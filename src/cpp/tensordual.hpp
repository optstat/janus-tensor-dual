#ifndef TENSORDUAL_H
#define TENSORDUAL_H

#include <torch/torch.h>
#include <type_traits> // For std::is_scalar
#include <vector>
#include <sstream>  // For std::ostringstream

using TensorIndex = torch::indexing::TensorIndex;
using Slice = torch::indexing::Slice;

void safe_update(
    torch::Tensor& tensor_to_update,
    const torch::Tensor& indices,
    const torch::Tensor& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    const torch::Tensor& indices,
    const double& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    const torch::Tensor& indices,
    const bool& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    const torch::Tensor& indices,
    const int& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}



void safe_update(
    torch::Tensor& tensor_to_update,
    std::vector<at::indexing::TensorIndex> indices,
    const torch::Tensor& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    at::indexing::TensorIndex indices,
    const torch::Tensor& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    at::indexing::TensorIndex indices,
    const double& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}



void safe_update(
    torch::Tensor& tensor_to_update,
    std::vector<at::indexing::TensorIndex> indices,
    const double& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    std::vector<at::indexing::TensorIndex> indices,
    const bool& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}

void safe_update(
    torch::Tensor& tensor_to_update,
    std::vector<at::indexing::TensorIndex> indices,
    const int& value) 
{
    auto tensor_copy = tensor_to_update.clone();
    tensor_copy.index_put_({indices}, value);
    tensor_to_update = tensor_copy;
}



namespace janus {

//Forward declaration of the TensorMatDual class
class TensorMatDual;

//Forward declaration of the TensorMatHyperDual class
class TensorMatHyperDual;

/**
 * @brief The TensorDual class
 * The TensorDual class encapsulates a real tensor and a dual tensor
 * and provides methods for automatic differentiation using dual numbers
 * while enforcing strict dimensional compatibility between the real and dual tensors
 * by limiting the real part to a vector and the dual part to a matrix.
 * Limiting the dual tensors in this way allows for effient vectorized operations.
 * Note that the goal of this class is not necessarily speed of computation,
 * but as low a memory footprint as possible while still allowing for efficient
 * vectorized operations.  This in turn allows for parallelization of the
 * operations on the dual part of the tensor especially when using GPUs.
 */
class TensorDual {
public:
    // Members
    torch::Tensor r;
    torch::Tensor d;
    torch::Dtype dtype;
    torch::Device device_ = torch::kCPU;

public:
    // Default constructor
    TensorDual() noexcept
        : r(torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
          d(torch::empty({0, 0, 0}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
          dtype(torch::kFloat64),
          device_(torch::kCPU) {}

    // Constructor by reference
    TensorDual(const torch::Tensor& r, const torch::Tensor& d) {
        if (r.dim() != 2) {
            throw std::invalid_argument("The real tensor `r` must be 2D.");
        }
        if (d.dim() != 3) {
            throw std::invalid_argument("The dual tensor `d` must be 3D.");
        }
        if (r.device() != d.device()) {
            throw std::invalid_argument("Real and dual tensors must reside on the same device.");
        }
        if (r.dtype() != d.dtype()) {
            throw std::invalid_argument("Real and dual tensors must have the same dtype.");
        }
        this->r = r;
        this->d = d;
        this->dtype = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }

    // Move constructor
    TensorDual(TensorDual&& other) noexcept
        : r(std::move(other.r)), d(std::move(other.d)), dtype(other.dtype), device_(other.device_) {}

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
            throw std::invalid_argument("Shape mismatch: grad.r must match r in size.");
        }
        if (grad.d.sizes() != d.sizes()) {
            throw std::invalid_argument("Shape mismatch: grad.d must match d in size.");
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
     * @throws std::invalid_argument If `r` has fewer than 2 dimensions.
     */
    static TensorDual create(const torch::Tensor& r) {
        // Ensure the input tensor has at least 2 dimensions
        if (r.dim() < 2) {
            throw std::invalid_argument("Input tensor `r` must have at least 2 dimensions.");
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
        // Ensure tensors are defined
        if (!r.defined() || !d.defined()) {
            throw std::runtime_error("Cannot create zeros_like: Tensors r and d must be defined.");
        }

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
        if (!first.r.defined() || !first.d.defined() || !second.r.defined() || !second.d.defined()) {
            throw std::invalid_argument("Both TensorDual objects must have defined real and dual tensors.");
        }

        // Validate einsum string
        auto pos = arg.find("->");
        if (pos == std::string::npos) {
            throw std::invalid_argument("Einsum string must contain '->'.");
        }

        // Compute the real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Compute the dual part
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 2);
        auto darg = arg1 + "z->" + arg2 + "z";

        auto d1 = torch::einsum(darg, {first.r, second.d});
        auto d2 = torch::einsum(darg, {second.r, first.d});

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
     * @brief Compute the sum of a TensorDual object along a specified dimension.
     * 
     * This static method computes the sum along the specified dimension for both the real (`r`)
     * and dual (`d`) parts of the TensorDual object. The result is returned as a new TensorDual object.
     * 
     * @param x The input TensorDual object (must have defined real and dual tensors).
     * @param dim The dimension along which to compute the sum (default is 1).
     * @return A new TensorDual object with the sum computed along the specified dimension.
     * @throws std::invalid_argument If the input TensorDual is not well-formed.
     */
    static TensorDual sum(const TensorDual& x, int64_t dim = 1) {
        // Validate input TensorDual
        if (!x.r.defined() || !x.d.defined()) {
            throw std::invalid_argument("Input TensorDual must have defined real and dual tensors.");
        }

        // Compute the sum along the specified dimension
        auto r = torch::sum(x.r, /*dim=*/dim, /*keepdim=*/true);
        auto d = torch::sum(x.d, /*dim=*/dim, /*keepdim=*/true);

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
     * @brief Compute the sum of the TensorDual object along a specified dimension.
     * 
     * This method computes the sum along the specified dimension for both the real (`r`) and dual (`d`)
     * parts of the TensorDual object. The result retains the reduced dimension using `unsqueeze`.
     * 
     * @param dim The dimension along which to compute the sum (default is 1).
     * @return A new TensorDual object with the sum computed along the specified dimension.
     * @throws std::invalid_argument If the TensorDual object is not well-formed.
     */
    TensorDual sum(int64_t dim = 1) {
        // Validate input tensors
        if (!r.defined() || !d.defined()) {
            throw std::invalid_argument("TensorDual object must have defined real and dual tensors.");
        }

        // Compute the sum along the specified dimension and retain the dimension
        auto real = torch::sum(r, /*dim=*/dim, /*keepdim=*/true);
        auto dual = torch::sum(d, /*dim=*/dim, /*keepdim=*/true);

        // Return the result as a new TensorDual
        return TensorDual(real, dual);
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
     * @brief Overload the addition operator for TensorDual and torch::Tensor.
     * 
     * This operator adds a torch::Tensor to the real part (`r`) of the TensorDual object
     * while leaving the dual part (`d`) unchanged. It creates a new TensorDual object as the result.
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

        // Perform addition on the real part
        return TensorDual(r + other, d.clone()); // Clone ensures dual part remains immutable
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

        // Check broadcast compatibility
        if (!are_broadcast_compatible(r.sizes(), other.sizes())) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must be broadcast-compatible with the real part of the TensorDual.");
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
        // Validate tensor dimensions
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Tensors in TensorDual must have the same shape for division.");
        }

        // Ensure the denominator is safe for division
        auto safe_r = other.r.clamp_min(1e-12);

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
        auto safe_other = othere.clamp_min(1e-12);

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
     * @throws std::invalid_argument If the scalar value is zero.
     */
    TensorDual operator/(double scalar) {
        // Check for division by zero
        if (scalar == 0.0) {
            throw std::invalid_argument("Division by zero is undefined.");
        }

        // Perform element-wise division
        return TensorDual(this->r / scalar, this->d / scalar);
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
        if (index.dim() >= r.dim()) {
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
        if (index.dim() >= r.dim()) {
            throw std::invalid_argument("Index tensor dimensions are incompatible with the target tensor.");
        }

        // Validate source TensorDual dimensions
        if (r.sizes() != src.r.sizes() || d.sizes() != src.d.sizes()) {
            throw std::invalid_argument("Source TensorDual must have the same shape as the target TensorDual.");
        }

        // Expand the index tensor for the dual part
        auto index_expanded = index.unsqueeze(-1);

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
        auto r_safe = r.clamp_min(1e-12); // Prevent division by zero
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
        // Validate that all real values are >= 0
        if (!torch::all(r >= 0).item<bool>()) {
            throw std::domain_error("Square root is only defined for non-negative real numbers.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
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
        auto sign_r = torch::sign(r).unsqueeze(-1);

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
        auto sign_r = torch::sign(r);

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
        auto signed_log_r = torch::sign(r) * torch::log(torch::abs(r) + 1.0);

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
        auto r_sloginv = torch::sign(r) * (exp_abs_r - 1.0);

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
        // Extract the imaginary part of the real and dual tensors
        auto imag_r = torch::imag(r);
        auto imag_d = torch::imag(d);

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
        if (!mask.dtype().is_boolean()) {
            throw std::invalid_argument("Mask must be a boolean tensor.");
        }

        // Validate that the mask has the same shape as the real tensor
        if (mask.sizes() != r.sizes()) {
            throw std::invalid_argument("Mask must have the same shape as the real tensor.");
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
        if (!mask.dtype().is_boolean()) {
            throw std::invalid_argument("Mask must be a boolean tensor.");
        }

        // Validate that the mask matches the shape of the real tensor
        if (mask.sizes() != r.sizes()) {
            throw std::invalid_argument("Mask must have the same shape as the real tensor.");
        }

        // Validate that the value TensorDual has compatible shapes
        if (value.r.sizes() != r.sizes() || value.d.sizes() != d.sizes()) {
            throw std::invalid_argument("Shapes of value TensorDual components must match the target tensors.");
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
    void index_put_(const torch::indexing::TensorIndex& mask, const Scalar& value) {
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

        // Compute the dual part corresponding to the maximum real part
        auto max_d = torch::where(this->r > other.r, this->d, other.d);

        // Return the resulting TensorDual
        return TensorDual(max_r, max_d);
    }


    /**
     * @brief Computes the sign of the real part of a TensorDual object.
     * 
     * This method computes the sign of the real (`r`) part of the TensorDual object.
     * The dual (`d`) part is set to zero since the derivative of the sign function is undefined
     * at zero and zero elsewhere.
     * 
     * @return A new TensorDual object with the sign of the real part and zero dual part.
     */
    TensorDual sign() const {
        // Compute the sign of the real part
        auto sign_r = torch::sign(this->r);

        // Set the dual part to zero
        auto zero_d = torch::zeros_like(this->d);

        // Return the resulting TensorDual
        return TensorDual(sign_r, zero_d);
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




/**
 * The TensorHyperDual class is meant to keep track of the sensitivities to the initial conditions
 * of a function. It is a generalization of the TensorDual class, which keeps track of the first
 * order derivatives of a function. The TensorHyperDual class keeps track of the first and second
 * order derivatives of a function in a vectorized manner.  The class is designed to run in parallel
 * on vector parallel devices and follows the pytorch conventions for tensor operations.
 */
class TensorHyperDual {
public:
    torch::Tensor r;
    torch::Tensor d;
    torch::Tensor h;
    torch::Dtype dtype_ = torch::kFloat64;
    torch::Device device_ = torch::kCPU;

    TensorHyperDual(int r_dim1, int r_dim2, int d_dim1, int d_dim2, int d_dim3, int h_dim1, int h_dim2, int h_dim3, int h_dim4) 
    : r(torch::zeros({r_dim1, r_dim2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      d(torch::zeros({d_dim1, d_dim2, d_dim3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      h(torch::zeros({h_dim1, h_dim2, h_dim3, h_dim4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))) {}
    
    
    TensorHyperDual()
    : r(torch::zeros({1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      d(torch::zeros({1, 1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      h(torch::zeros({1, 1, 1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      dtype_(torch::kFloat64),
      device_(torch::kCPU) {}

    TensorHyperDual(torch::Tensor r, torch::Tensor d, torch::Tensor h) {
        validateTensors(r, d, h);
        this->r = r;
        this->d = d;
        this->h = h;
        dtype_ = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }

    TensorHyperDual to(torch::Device device) {
        this->r = this->r.to(device);
        this->d = this->d.to(device);
        this->h = this->h.to(device);
        this->device_ = device;
        return *this;
    }

    torch::Device device() const {
        return this->device_;
    }
private:
    void validateTensors(const torch::Tensor& r, const torch::Tensor& d, const torch::Tensor& h) const {
        if (r.dim() != 3) throw std::invalid_argument("Real part must be a 3D tensor");
        if (d.dim() != 4) throw std::invalid_argument("Dual part must be a 4D tensor");
        if (h.dim() != 5) throw std::invalid_argument("Hyperdual part must be a 5D tensor");
        if (r.device() != d.device() || d.device() != h.device()) {
            throw std::invalid_argument("All tensors must reside on the same device");
        }
        if (r.dtype() != d.dtype() || d.dtype() != h.dtype()) {
            throw std::invalid_argument("All tensors must have the same dtype");
        }
    }

public:

  /**
   * @brief Construct a new TensorHyperDual object from a TensorDual object.
   * 
   * The real and dual parts are the same. We construct a new TensorHyperDual 
   * object with the same real and dual parts but with a zero hyperdual part 
   * that has an additional dimension replicating the dual part's last dimension.
   */
    TensorHyperDual(const TensorDual& x)
      : r(x.r),
        d(x.d),
        h(torch::zeros({x.d.size(0), x.d.size(1), x.d.size(2), x.d.size(3), x.d.size(3)}, x.d.options())),
        dtype_(torch::typeMetaToScalarType(x.r.dtype())),
        device_(x.r.device()) {
        // Validate input dimensions and consistency
        if (x.r.dim() != 3) {
          throw std::invalid_argument("TensorDual real part must be a 3D tensor");
        }
        if (x.d.dim() != 4) {
          throw std::invalid_argument("TensorDual dual part must be a 4D tensor");
        }
        if (x.r.device() != x.d.device()) {
          throw std::invalid_argument("TensorDual real and dual parts must reside on the same device");
        }
        if (x.r.dtype() != x.d.dtype()) {
          throw std::invalid_argument("TensorDual real and dual parts must have the same dtype");
        }
    }

    /**
     * Shallow copy constructor
     */
    TensorHyperDual(const TensorHyperDual& other)
    : r(other.r),
      d(other.d),
      h(other.h),
      dtype_(other.dtype_),
      device_(other.device_) {}

    TensorHyperDual contiguous() const {
        // Ensure each tensor (r, d, h) is stored in contiguous memory
        return TensorHyperDual(r.contiguous(), d.contiguous(), h.contiguous());
    }

    TensorMatHyperDual eye();


    /**
     * Sum the TensorHyperDual along the specified dimension.
     * 
     * @param dim The dimension to sum along (default: 1).
     * @param keepdim Whether to retain reduced dimensions (default: true).
     * @return A new TensorHyperDual object with summed components.
     * 
     * @note Assumes that `r`, `d`, and `h` are tensors with compatible dimensions.
     */
    TensorHyperDual sum(int64_t dim = 1, bool keepdim=true) const {
        auto r = this->r.sum(dim, true);
        auto d = this->d.sum(dim, true);
        auto h = this->h.sum(dim, true);
        return TensorHyperDual(r, d, h);
    }
    
    
    /**
     * Compute the square of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object with squared components.
     *
     * The computation follows:
     * - r_new = r^2
     * - d_new = 2 * r * d
     * - h_new = 2 * d^2 + 2 * r * h
     */
    TensorHyperDual square() const {
      // Compute the square of the real part
      auto rsq = r.square();

      // Compute the dual part using the chain rule
      auto dn = 2 * torch::einsum("mi, mij->mij", {this->r, this->d});

      // Compute the hyperdual part using the chain rule
      auto hn = 2 * torch::einsum("mij, mik->mijk", {this->d, this->d}) +
              2 * torch::einsum("mi, mijk->mijk", {this->r, this->h});

      // Return the resulting TensorHyperDual object
      return TensorHyperDual(rsq, dn, hn);
    }

    /**
     * Compute the square root of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object with square root components.
     *
     * The computation follows:
     * - r_new = sqrt(r)
     * - d_new = d / (2 * sqrt(r))
     * - h_new = -d^2 / (4 * r^(3/2)) + h / (2 * sqrt(r))
     */
    TensorHyperDual sqrt() const {
      // Compute the square root of the real part
      auto rsq = r.sqrt();

      // Compute the dual part
      auto dn = 0.5 * torch::einsum("mi, mij->mij", {r.pow(-0.5), d});

      // Compute the hyperdual part
      auto hn = -0.25 * torch::einsum("mi, mij, mik->mijk", {r.pow(-1.5), d, d}) +
                0.5 * torch::einsum("mi, mijk->mijk", {r.pow(-0.5), h});

      // Return the resulting TensorHyperDual object
      return TensorHyperDual(rsq, dn, hn);
    }
    
    /**
     * Overload the addition operator for TensorHyperDual.
     *
     * @param other The TensorHyperDual object to add.
     * @return A new TensorHyperDual object representing the element-wise sum.
     */
    TensorHyperDual operator+(const TensorHyperDual& other) const {
      return TensorHyperDual(this->r + other.r, this->d + other.d, this->h + other.h);
    }
    
    /**
     * Overload the unary negation operator for TensorHyperDual.
     *
     * @return A new TensorHyperDual object with each component negated.
     */
    TensorHyperDual operator-() const {
      return TensorHyperDual(-r, -d, -h);
    }

    /**
     * Overload the subtraction operator for TensorHyperDual.
     *
     * @param other The TensorHyperDual object to subtract.
     * @return A new TensorHyperDual object representing the element-wise difference.
    */
    TensorHyperDual operator-(const TensorHyperDual& other) const {
      auto r_diff = this->r - other.r;
      auto d_diff = this->d - other.d;
      auto h_diff = this->h - other.h;
      return TensorHyperDual(r_diff, d_diff, h_diff);
    }
    /**
     * Overload the multiplication operator for TensorHyperDual.
     *
     * @param other The TensorHyperDual object to multiply with.
     * @return A new TensorHyperDual object representing the product.
     *
     * The computation follows:
     * - r_new = r1 * r2
     * - d_new = r1 * d2 + d1 * r2
     * - h_new = d1 * d2 + d2 * d1 + r1 * h2 + r2 * h1
     */
    TensorHyperDual operator*(const TensorHyperDual& other) const {
      // Real part
      auto rn = this->r * other.r;

      // First-order derivative (dual part)
      auto dn = torch::einsum("mi, mij->mij", {this->r, other.d})
            + torch::einsum("mi, mij->mij", {other.r, this->d});

      // Second-order derivative (hyperdual part)
      auto hn = torch::einsum("mij, mik->mijk", {this->d, other.d}) // d1 * d2
            + torch::einsum("mij, mik->mijk", {other.d, this->d}) // d2 * d1 (symmetric)
            + torch::einsum("mi, mijk->mijk", {this->r, other.h}) // r1 * h2
            + torch::einsum("mi, mijk->mijk", {other.r, this->h}); // r2 * h1

      // Return the resulting TensorHyperDual object
      return TensorHyperDual(rn, dn, hn);
    }
  /**
   * Overload the multiplication operator for TensorHyperDual.
   *
   * @param other The TensorHyperDual object to multiply with.
   * @return A new TensorHyperDual object representing the product.
   *
   * The computation follows:
   * - r_new = r1 * r2
   * - d_new = r1 * d2 + d1 * r2
   * - h_new = d1 * d2 + d2 * d1 + r1 * h2 + r2 * h1
   */
  TensorHyperDual operator*(const TensorHyperDual& other) const {
    // Real part
    auto rn = this->r * other.r;

    // First-order derivative (dual part)
    auto dn = torch::einsum("mi, mij->mij", {this->r, other.d})
            + torch::einsum("mi, mij->mij", {other.r, this->d});

    // Second-order derivative (hyperdual part)
    auto hn = torch::einsum("mij, mik->mijk", {this->d, other.d}) // d1 * d2 outer product
            + torch::einsum("mij, mik->mijk", {other.d, this->d}) // d2 * d1 (symmetric)
            + torch::einsum("mi, mijk->mijk", {this->r, other.h}) // r1 * h2
            + torch::einsum("mi, mijk->mijk", {other.r, this->h}); // r2 * h1

    // Return the resulting TensorHyperDual object
    return TensorHyperDual(rn, dn, hn);
  }

  /**
   * Overload the division operator for TensorHyperDual.
   *
   * @param other The TensorHyperDual object to divide by.
   * @return A new TensorHyperDual object representing the quotient.
   *
   * The computation follows:
   * - r_new = r1 / r2
   * - d_new = (d1 / r2) - (r1 * d2) / r2^2
   * - h_new = h1 / r2 - 2 * (d1 * d2) / r2^2 + 2 * (r1 * d2^2) / r2^3 - (r1 * h2) / r2^2
   */
  TensorHyperDual operator/(const TensorHyperDual& other) const {
    // Real part
    auto rn = this->r / other.r;

    // First-order derivative
    auto dn = (this->d / other.r.unsqueeze(-1)) 
              - (this->r / other.r.pow(2)).unsqueeze(-1) * other.d;

    // Second-order derivative
    auto hn = torch::einsum("mijk, mi->mijk", {this->h, other.r.reciprocal()})  // h1 / r2
            - 2 * torch::einsum("mij, mik->mijk", {this->d / other.r.unsqueeze(-1), other.d / other.r.unsqueeze(-1)})  // (d1 * d2) / r2^2
            + 2 * torch::einsum("mi, mij, mik->mijk", {this->r / other.r.pow(3), other.d, other.d})  // (r1 * d2^2) / r2^3
            - torch::einsum("mi, mijk->mijk", {this->r / other.r.pow(2), other.h});  // (r1 * h2) / r2^2

    // Return the resulting TensorHyperDual object
    return TensorHyperDual(rn, dn, hn);
  }



   /**
    * Overload the division operator for TensorHyperDual when dividing by a simple tensor.
    *
    * @param other A simple torch::Tensor with only a real part (no dual or hyperdual components).
    * @return A new TensorHyperDual object representing the quotient.
    *
    * The computation follows:
    * - r_new = r1 / r2
    * - d_new = d1 / r2
    * - h_new = h1 / r2
    */
   TensorHyperDual operator/(const torch::Tensor& other) const {
    // Ensure `other` has the same dimensionality as the real part `r`
    auto othere = other.dim() != this->r.dim() ? other.unsqueeze(1) : other;

    // Real part
    auto rn = this->r / othere;

    // First-order derivative (dual part)
    auto dn = this->d / othere.unsqueeze(-1);

    // Second-order derivative (hyperdual part)
    auto hn = torch::einsum("mijk, mi->mijk", {this->h, othere.reciprocal()});

    // Return the resulting TensorHyperDual object
    return TensorHyperDual(rn, dn, hn); 
   }

   /**
    * Overload the division operator for TensorHyperDual by a scalar.
    *
    * @param scalar A scalar value to divide each component by.
    * @return A new TensorHyperDual object with each component divided by the scalar.
    *
    * @throws std::runtime_error if scalar is zero.
    */
    TensorHyperDual operator/(const double& scalar) const {
        // Check for division by zero
        TORCH_CHECK(scalar != 0, "Division by zero is not allowed.");

        // Perform element-wise division
        return TensorHyperDual(this->r / scalar, this->d / scalar, this->h / scalar);
    }
    /**
     * Overload the less-than-or-equal-to operator for TensorHyperDual <= TensorHyperDual.
     *
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating which batch elements satisfy the condition.
     *
     * Note: This operator compares only the real part (r) of the TensorHyperDual objects.
     */
    torch::Tensor operator<=(const TensorHyperDual& other) const {
        // Perform element-wise comparison of the real parts
        auto mask = r <= other.r;

        // Reduce across non-batch dimensions (if necessary) to generate a batch-level mask
        auto batch_mask = mask.all(1); // Adjust dimension as needed based on tensor shape

        // Return the mask for batch selection
        return batch_mask;
    }


    /**
     * Overload the equality operator for TensorHyperDual == TensorHyperDual (batch-level).
     *
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask of size [B], where each entry indicates whether
     *         all elements in the corresponding batch of the real part (r) are equal.
     *
     * Note: This operator only compares the real part (r) of the TensorHyperDual objects.
     */
    torch::Tensor operator==(const TensorHyperDual& other) const {
        // Perform element-wise comparison of the real parts
        auto mask = r == other.r;

        // Reduce across the feature dimension (N) to get a batch-level mask
        auto batch_mask = mask.all(1);

        return batch_mask;
    }

    /**
     * Overload the less-than-or-equal-to operator for TensorHyperDual <= torch::Tensor.
     * This also implicitly supports TensorHyperDual <= Scalar due to PyTorch's implicit scalar handling.
     *
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask of the same shape as the broadcasted tensors.
     *
     * Note: This operator only compares the real part (r) of the TensorHyperDual object.
     */
    torch::Tensor operator<=(const torch::Tensor& other) const {
        // Perform element-wise comparison of the real part with the other tensor
        auto mask = r <= other;

        // Return the mask directly (no squeezing required)
        auto batch_mask = mask.all(1);

        return batch_mask;

    }


    /**
     * Overload the less-than-or-equal-to operator for TensorHyperDual <= Scalar.
     *
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating the comparison result.
     */
    template <typename Scalar>
    torch::Tensor operator<=(const Scalar& scalar) const {
        // Perform element-wise comparison of the real part with the scalar
        auto mask = (r <= scalar);

        //Remove the last dimension
        auto batch_mask = mask.all(1);
        return batch_mask;
    }


    /**
     * Overload the greater-than operator for TensorHyperDual > TensorHyperDual.
     *
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     *         is greater than the real part of the other TensorHyperDual.
     *
     * Note: This operator only compares the real part (r) of the TensorHyperDual objects.
     */
    torch::Tensor operator>(const TensorHyperDual& other) const {
        // Perform element-wise comparison of the real parts
        auto mask = r > other.r;

        //Remove the last dimension
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the greater-than operator for TensorHyperDual > torch::Tensor.
     * This also implicitly supports TensorHyperDual > Scalar due to PyTorch's implicit scalar handling.
     *
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     *         is greater than the other tensor.
     *
     * Note: This operator only compares the real part (r) of the TensorHyperDual object.
     */
    torch::Tensor operator>(const torch::Tensor& other) const {
        auto mask = r > other;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the greater-than operator for TensorHyperDual > Scalar.
     *
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     *         is greater than the scalar value.
     */
    template <typename Scalar>
    torch::Tensor operator>(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        auto mask = r > scalar_tensor;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the less-than operator for TensorHyperDual < TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator<(const TensorHyperDual& other) const {
        auto mask = r < other.r;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the less-than operator for TensorHyperDual < torch::Tensor.
     * This also implicitly supports TensorHyperDual < Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator<(const torch::Tensor& other) const {
        auto mask = r < other;
        //std::cerr << "mask sizes in <: " << mask.sizes() << std::endl;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the less-than operator for TensorHyperDual < Scalar.
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    template <typename Scalar>
    torch::Tensor operator<(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r < scalar;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }



    /**
     * Overload the greater-than-or-equal-to operator for TensorHyperDual >= TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator>=(const TensorHyperDual& other) const {
        auto mask = r >= other.r;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the greater-than-or-equal-to operator for TensorHyperDual >= torch::Tensor.
     * This also implicitly supports TensorHyperDual >= Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator>=(const torch::Tensor& other) const {
        auto mask = r >= other;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the greater-than-or-equal-to operator for TensorHyperDual >= Scalar.
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    template <typename Scalar>
    torch::Tensor operator>=(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        auto mask = r >= scalar_tensor;
        auto batch_mask = mask.all(1);
        return batch_mask;
    }



    /**
     * Overload the equality operator for TensorHyperDual == torch::Tensor.
     * This also implicitly supports TensorHyperDual == Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator==(const torch::Tensor& other) const {
        auto mask = r.eq(other); // Using the .eq() function for equality comparison
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    
    /**
     * Overload the equality operator for TensorHyperDual == Scalar.
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    template <typename Scalar>
    torch::Tensor operator==(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r.eq(scalar);
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the inequality operator for TensorHyperDual != TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator!=(const TensorHyperDual& other) const {
        auto mask = r.ne(other.r); // Using the .ne() function for inequality comparison
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Overload the inequality operator for TensorHyperDual != torch::Tensor.
     * This also implicitly supports TensorHyperDual != Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator!=(const torch::Tensor& other) const {
        auto mask = r.ne(other); // Using the .ne() function for inequality comparison
        auto batch_mask = mask.all(1);
        return batch_mask;
    }
    

    /**
     * Overload the inequality operator for TensorHyperDual != Scalar.
     * @tparam Scalar A scalar type convertible to Tensor (e.g., int, float, double).
     * @param scalar A scalar value to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    template <typename Scalar>
    torch::Tensor operator!=(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r.ne(scalar);
        auto batch_mask = mask.all(1);
        return batch_mask;
    }

    /**
     * Compute the reciprocal of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the reciprocal.
     */
    TensorHyperDual reciprocal() const {
        // Real part
        auto rrec = this->r.reciprocal();  // 1 / r
        auto rrec_sq = rrec * rrec;        // (1 / r)^2
        auto rrec_cube = rrec_sq * rrec;   // (1 / r)^3

        // First-order derivative
        auto dn = -rrec_sq.unsqueeze(-1) * this->d;

        // Second-order derivative
        auto hn = 2 * torch::einsum("mi, mij, mik->mijk", {rrec_cube, this->d, this->d}) -
                torch::einsum("mi, mijk->mijk", {rrec_sq, this->h});

        return TensorHyperDual(rrec, dn, hn);
    }    
    /**
     * Compute the cosine of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the cosine.
     */
    TensorHyperDual cos() const {
        // Real part
        auto rn = torch::cos(this->r);  // Compute cos(r)

        // First-order derivative
        auto dn = -torch::sin(this->r).unsqueeze(-1) * this->d;  // -sin(r) * d

        // Second-order derivative
        auto hn = -torch::einsum("mi, mij, mik->mijk", {torch::cos(this->r), this->d, this->d}) -
                torch::einsum("mi, mijk->mijk", {torch::sin(this->r), this->h});  // -cos(r)*d*d - sin(r)*h

        return TensorHyperDual(rn, dn, hn);
    }
    /**
     * Compute the sine of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the sine.
     */
    TensorHyperDual sin() const {
        // Real part
        auto rn = torch::sin(this->r);  // Compute sin(r)

        // First-order derivative
        auto dn = torch::cos(this->r).unsqueeze(-1) * this->d;  // cos(r) * d

        // Second-order derivative
        auto hn = -torch::einsum("mi, mij, mik->mijk", {torch::sin(this->r), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::cos(this->r), this->h});  // -sin(r)*d*d + cos(r)*h

        return TensorHyperDual(rn, dn, hn);
    }
    /**
     * Compute the tangent of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the tangent.
     */
    TensorHyperDual tan() const {
        // Real part
        auto rn = torch::tan(this->r);  // Compute tan(r)

        // First-order derivative
        auto dn = torch::pow(torch::cos(this->r), -2).unsqueeze(-1) * this->d;  // sec^2(r) * d

        // Second-order derivative
        auto hn = 2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cos(this->r), -2) * torch::tan(this->r), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::pow(torch::cos(this->r), -2), this->h});  // 2*sec^2(r)*tan(r)*d*d + sec^2(r)*h

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * Compute the inverse sine of the TensorHyperDual object.
     * 
     */
    TensorHyperDual asin() const {
        auto one_minus_r_sq = 1 - this->r * this->r;  // 1 - r^2
        auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);  // sqrt(1 - r^2)

        // Real part
        auto rn = torch::asin(this->r);

        // First-order derivative
        auto dn = (1 / sqrt_one_minus_r_sq).unsqueeze(-1) * this->d;

        // Second-order derivative
        auto hn = torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * this->r, this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), this->h});

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * Compute the inverse cosine of the TensorHyperDual object.
     */
    TensorHyperDual acos() const {
        auto one_minus_r_sq = 1 - this->r * this->r;  // 1 - r^2
        auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);  // sqrt(1 - r^2)

        // Real part
        auto rn = torch::acos(this->r);

        // First-order derivative
        auto dn = -(1 / sqrt_one_minus_r_sq).unsqueeze(-1) * this->d;

        // Second-order derivative
        auto hn = -torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * this->r, this->d, this->d}) -
                torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), this->h});

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * Compute the inverse tangent of the TensorHyperDual object.
     */
    TensorHyperDual atan() const {
        auto one_plus_r_sq = 1 + this->r * this->r;  // 1 + r^2

        // Real part
        auto rn = torch::atan(this->r);

        // First-order derivative
        auto dn = (1 / one_plus_r_sq).unsqueeze(-1) * this->d;

        // Second-order derivative
        auto hn = -torch::einsum("mi, mij, mik->mijk", {(2 / torch::pow(one_plus_r_sq, 2)) * this->r, this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {(1 / one_plus_r_sq), this->h});

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * Compute the hyperbolic sine of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the hyperbolic sine.
     */
    TensorHyperDual sinh() const {
        // Real part
        auto rn = torch::sinh(this->r);  // Compute sinh(r)

        // First-order derivative
        auto dn = torch::cosh(this->r).unsqueeze(-1) * this->d;  // cosh(r) * d

        // Second-order derivative
        auto hn = torch::einsum("mi, mij, mik->mijk", {torch::sinh(this->r), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::cosh(this->r), this->h});  // sinh(r)*d*d + cosh(r)*h

        return TensorHyperDual(rn, dn, hn);
    }
    /**
     * Compute the hyperbolic cosine of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the hyperbolic cosine.
     */
    TensorHyperDual cosh() const {
        // Real part
        auto rn = torch::cosh(this->r);  // Compute cosh(r)

        // First-order derivative
        auto dn = torch::sinh(this->r).unsqueeze(-1) * this->d;  // sinh(r) * d

        // Second-order derivative
        auto hn = torch::einsum("mi, mij, mik->mijk", {torch::cosh(this->r), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::sinh(this->r), this->h});  // cosh(r)*d*d + sinh(r)*h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * Compute the hyperbolic tangent of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the hyperbolic tangent.
     */
    TensorHyperDual tanh() const {
        // Real part
        auto rn = torch::tanh(this->r);  // Compute tanh(r)

        // First-order derivative
        auto dn = torch::pow(torch::cosh(this->r), -2).unsqueeze(-1) * this->d;  // sech^2(r) * d

        // Second-order derivative
        auto hn = -2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cosh(this->r), -2) * torch::tanh(this->r), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::pow(torch::cosh(this->r), -2), this->h});  // -2*sech^2(r)*tanh(r)*d*d + sech^2(r)*h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * Compute the exponential of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the exponential.
     */
    TensorHyperDual exp() const {
        // Real part
        auto rn = torch::exp(this->r);  // Compute exp(r)

        // First-order derivative
        auto dn = rn.unsqueeze(-1) * this->d;  // exp(r) * d

        // Second-order derivative
        auto hn = torch::einsum("mi, mij, mik->mijk", {rn, this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {rn, this->h});  // exp(r) * (d * d + h)

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * Compute the natural logarithm of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the natural logarithm.
     */
    TensorHyperDual log() const {
        // Real part
        auto rn = torch::log(this->r);  // Compute log(r)

        // First-order derivative
        auto dn = torch::pow(this->r, -1).unsqueeze(-1) * this->d;  // 1 / r * d

        // Second-order derivative
        auto hn = -torch::einsum("mi, mij, mik->mijk", {torch::pow(this->r, -2), this->d, this->d}) +
                torch::einsum("mi, mijk->mijk", {torch::pow(this->r, -1), this->h});  // -1 / r^2 * d * d + 1 / r * h

        return TensorHyperDual(rn, dn, hn);
    }


    /**
     * Compute the absolute value of the TensorHyperDual object.
     *
     * @return A new TensorHyperDual object representing the absolute value.
     */
    TensorHyperDual abs() const {
        // Real part
        auto abs_r = torch::abs(this->r);  // Compute |r|

        // First-order derivative
        auto sign_r = torch::sign(this->r);  // Compute sign(r)
        auto dn = sign_r.unsqueeze(-1) * this->d;  // sign(r) * d

        // Second-order derivative
        auto hn = torch::einsum("mi, mijk->mijk", {sign_r, this->h});  // sign(r) * h

        return TensorHyperDual(abs_r, dn, hn);
    }

    /**
     * Convert the TensorHyperDual object to a complex TensorHyperDual object.
     */
    TensorHyperDual complex() const{
        torch::Tensor rc, dc, hc;
        this->r.is_complex() ? rc = this->r : rc = torch::complex(this->r, torch::zeros_like(this->r)).to(this->r.device());
        this->d.is_complex() ? dc = this->d : dc = torch::complex(this->d, torch::zeros_like(this->d)).to(this->d.device());
        this->h.is_complex() ? hc = this->h : hc = torch::complex(this->h, torch::zeros_like(this->h)).to(this->h.device());
        return TensorHyperDual(std::move(rc), std::move(dc), std::move(hc));
    }

    /**
     * Extract the real part of the TensorHyperDual object.
     */
    TensorHyperDual real() const {
        auto r = torch::real(this->r);
        auto d = torch::real(this->d);
        auto h = torch::real(this->h);
        return TensorHyperDual(std::move(r), std::move(d), std::move(h));
    }   

    /**
     * Extract the imaginary part of the TensorHyperDual object.
     */
    TensorHyperDual imag() {
        auto r = torch::imag(this->r);
        auto d = torch::imag(this->d);
        auto h = torch::imag(this->h);
        return TensorHyperDual(r, d, h);
    }

    /**
     * Create a TensorHyperDual object with the same shape as the current object,
     * but with all values set to zero.
     *
     * @return A new TensorHyperDual object where all components (r, d, h) are zeros.
     */
    TensorHyperDual zeros_like() const {
        return TensorHyperDual(
            torch::zeros_like(this->r),  // Zeros for the real part
            torch::zeros_like(this->d),  // Zeros for the dual part
            torch::zeros_like(this->h)   // Zeros for the hyperdual part
        );
    }

    /**
     * Create a TensorHyperDual object with the same shape as the given object,
     * but with all values set to zero.
     *
     * @param other The TensorHyperDual object whose shape and type are used
     *              to create the new zeros-like TensorHyperDual.
     * @return A new TensorHyperDual object where all components (r, d, h) are zeros.
     */
    static TensorHyperDual zeros_like(const TensorHyperDual& other) {
        return TensorHyperDual(
            torch::zeros_like(other.r),  // Zeros for the real part
            torch::zeros_like(other.d),  // Zeros for the dual part
            torch::zeros_like(other.h)   // Zeros for the hyperdual part
        );
    }

    /**
     * Compute the sign of the TensorHyperDual object.
     * The derivative of the sign function is zero everywhere, so the dual and hyperdual parts are zero.
     *
     * @return A new TensorHyperDual object representing the sign.
     */
    TensorHyperDual sign() const {
        // Compute the sign of the real part
        auto sign_r = torch::sign(this->r);  // sign(r)

        // Dual and hyperdual parts are zero
        auto sign_d = torch::zeros_like(this->d);  // 0 for the dual part
        auto sign_h = torch::zeros_like(this->h);  // 0 for the hyperdual part

        return TensorHyperDual(sign_r, sign_d, sign_h);
    }
    
    /**
     * Compute the minimum value of the TensorHyperDual object along dimension 1.
     */
    TensorHyperDual min() {
        // Compute the min values and indices along dimension 1, keeping the dimension
        auto min_result = torch::min(this->r, /*dim=*/1, /*keepdim=*/true);
        auto min_values = std::get<0>(min_result);  // Minimum values
        auto min_indices = std::get<1>(min_result); // Indices of the minimum values

        // Validate tensor dimensions for consistency
        if (this->d.size(1) != this->r.size(1) || this->h.size(1) != this->r.size(1)) {
            throw std::invalid_argument("Shape mismatch: `d` and `h` tensors must match the size of `r` along dimension 1.");
        }

        // Adjust the shape of min_indices to match the dual tensor for gathering
        auto dshape = min_indices.unsqueeze(-1).expand({
            min_indices.size(0), 
            min_indices.size(1), 
            this->d.size(-1)
        });

        // Adjust the shape of min_indices to match the hyperdual tensor for gathering
        auto hshape = min_indices.unsqueeze(-1).unsqueeze(-1).expand({
            min_indices.size(0), 
            min_indices.size(1), 
            this->h.size(-2), 
            this->h.size(-1)
        });

        // Gather the dual and hyperdual values based on the min indices
        auto dual_values = torch::gather(this->d, /*dim=*/1, dshape);
        auto hyper_values = torch::gather(this->h, /*dim=*/1, hshape);

        // Construct and return a new TensorHyperDual with the min values and corresponding dual and hyperdual values
        return TensorHyperDual(min_values, dual_values, hyper_values);
    }
    
    /**
     * Compute the maximum value of the TensorHyperDual object along dimension 1.
     */
    TensorHyperDual max() {
        // Compute the max values and indices along dimension 1, keeping the dimension
        auto max_result = torch::max(this->r, /*dim=*/1, /*keepdim=*/true);
        auto max_values = std::get<0>(max_result);  // Maximum values
        auto max_indices = std::get<1>(max_result); // Indices of the maximum values

        // Validate tensor dimensions for consistency
        if (this->d.size(1) != this->r.size(1) || this->h.size(1) != this->r.size(1)) {
            throw std::invalid_argument("Shape mismatch: `d` and `h` tensors must match the size of `r` along dimension 1.");
        }

        // Adjust the shape of max_indices to match the dual tensor for gathering
        auto dshape = max_indices.unsqueeze(-1).expand({
            max_indices.size(0), 
            max_indices.size(1), 
            this->d.size(-1)
        });

        // Adjust the shape of max_indices to match the hyperdual tensor for gathering
        auto hshape = max_indices.unsqueeze(-1).unsqueeze(-1).expand({
            max_indices.size(0), 
            max_indices.size(1), 
            this->h.size(-2), 
            this->h.size(-1)
        });

        // Gather the dual and hyperdual values based on the max indices
        auto dual_values = torch::gather(this->d, /*dim=*/1, dshape);
        auto hyper_values = torch::gather(this->h, /*dim=*/1, hshape);

        // Construct and return a new TensorHyperDual with the max values and corresponding dual and hyperdual values
        return TensorHyperDual(max_values, dual_values, hyper_values);
    }

    /**
     * Implement the where function for TensorHyperDual objects.
     */
    static TensorHyperDual where(const torch::Tensor& cond, 
                                 const TensorHyperDual& x, 
                                 const TensorHyperDual& y) {
        // Ensure that the shapes of x and y match
        if (x.r.sizes() != y.r.sizes() || x.d.sizes() != y.d.sizes() || x.h.sizes() != y.h.sizes()) {
            throw std::invalid_argument("Tensor shapes of x and y must match for TensorHyperDual::where.");
        }

        // Create condition tensors for r, d, and h
        torch::Tensor condr, condd, condh;

        if (cond.dim() == 1) {
            // Expand cond for each dimension of x and y
            condr = cond.unsqueeze(1).expand({-1, x.r.size(1)});
            condd = cond.unsqueeze(1).unsqueeze(2).expand({-1, x.d.size(1), x.d.size(2)});
            condh = cond.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand({-1, x.h.size(1), x.h.size(2), x.h.size(3)});
        } else if (cond.sizes() == x.r.sizes()) {
            // Directly use cond if its shape matches x.r
            condr = cond;
            condd = cond.unsqueeze(2).expand({-1, -1, x.d.size(2)});
            condh = cond.unsqueeze(2).unsqueeze(3).expand({-1, -1, x.h.size(2), x.h.size(3)});
        } else {
            throw std::invalid_argument("Shape of cond must match x.r or be broadcastable to its shape.");
        }

        // Perform element-wise selection using torch::where
        auto xr = torch::where(condr, x.r, y.r);
        auto xd = torch::where(condd, x.d, y.d);
        auto xh = torch::where(condh, x.h, y.h);

        // Return a new TensorHyperDual object with selected values
        return TensorHyperDual(xr, xd, xh);
    }

    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     */
    static TensorHyperDual einsum(const std::string& arg, TensorHyperDual& first, TensorHyperDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos1  = arg.find(",");
        auto pos2  = arg.find("->");
        auto arg1  = arg.substr(0, pos1);
        auto arg2  = arg.substr(pos1+1, pos2-pos1-1);
        auto arg3  = arg.substr(pos2+2);

        // Find the position of the '->' in the einsum string
        auto r1 = first.r;
        auto r2 = second.r;
        auto d1 = first.d;
        auto d2 = second.d;
        auto h1 = first.h;
        auto h2 = second.h;

        auto r1d2arg = arg1+","+arg2+"z->"+arg3+"z";
        auto r1d2 = torch::einsum(r1d2arg, {r1, d2});
        auto d1r2arg = arg1+"z,"+arg2+"->"+arg3+"z";
        auto d1r2 = torch::einsum(d1r2arg, {d1, r2});
        auto d = r1d2 + d1r2;
        //Outer product
        auto d1d2arg = arg1+"z,"+arg2+"w->"+arg3+"zw";
        auto d1d2 = torch::einsum(d1d2arg, {d1, d2});
        auto r1h2arg = arg1+","+arg2+"zw->"+arg3+"zw";
        auto r1h2 = torch::einsum(r1h2arg, {r1, h2});
        auto h1r2arg = arg1+"zw,"+arg2+"->"+arg3+"zw";
        auto h1r2 = torch::einsum(h1r2arg, {h1, r2});
        auto h = r1h2 + 2*d1d2+ h1r2;

        return TensorHyperDual(std::move(r), std::move(d), std::move(h));
    }



    static TensorHyperDual einsum(const std::string& arg, 
                                  const torch::Tensor& first, 
                                  const TensorHyperDual & second)
    {

        auto r = torch::einsum(arg, {first, second.r});

        // Find the position of the '->' in the einsum string
        auto pos1  = arg.find(",");
        auto pos2  = arg.find("->");
        auto arg1  = arg.substr(0, pos1);
        auto arg2  = arg.substr(pos1+1, pos2-pos1-1);
        auto arg3  = arg.substr(pos2+2);
        
        //This is commutative so we can put the dual part at the end
        auto r1d2arg = arg1+","+arg2+"z->"+arg3+"z";
        auto r1d2 = torch::einsum(r1d2arg, {first, second.d});
        auto d = r1d2;

        auto r1h2arg = arg1+","+arg2+"zw->"+arg3+"zw";
        auto r1h2 = torch::einsum(r1h2arg, {first, second.h});
        auto h = r1h2;
        return TensorHyperDual(r, d, h);
        
    }


    static TensorHyperDual einsum(const std::string& arg, 
                                  const TensorHyperDual& first, 
                                  const torch::Tensor & second)
    {
        auto r = torch::einsum(arg, {first.r, second});

        // Find the position of the '->' in the einsum string
        auto pos1 = arg.find(",");
        auto pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos1);
        auto arg2 = arg.substr(pos1+1,pos2-pos1-1);
        auto arg3 = arg.substr(pos2+2);
        auto r1 = first.r;
        auto d1 = first.d;
        auto h1 = first.h;
        auto r2 = second;
        auto d1r2arg = arg1+"z,"+arg2+"->"+arg3+"z";
        auto d1r2 = torch::einsum(d1r2arg, {d1, r2});
        auto d = d1r2;

        auto h1r2arg = arg1+"zw,"+arg2+"->"+arg3+"zw";
        auto h1r2 = torch::einsum(h1r2arg, {h1, r2});
        auto d1d2arg = arg1+"z,"+arg2+"w->"+arg3+"zw";
        auto d1d2 = torch::einsum(d1d2arg, {d1, r2});
        auto h = h1r2 + d1d2;

        return TensorHyperDual(r, d, h);
    }


    static TensorHyperDual einsum(const std::string& arg, 
                             std::vector<TensorHyperDual> tensors) 
    {
        assert (arg.find("->") != std::string::npos && "einsum string must contain '->'");
        assert (arg.find(",") != std::string::npos && "einsum string must contain ','");
        assert (arg.find("z")== std::string::npos && "z is a reserved character in einsum used to operate on dual numbers");
        assert (arg.find("w")== std::string::npos && "w is a reserved character in einsum used to operate on dual numbers");
        std::vector<torch::Tensor> r_tensors;
        std::vector<torch::Tensor> d_tensors;
        std::vector<torch::Tensor> h_tensors;
        for (const auto& t : tensors) {
            r_tensors.push_back(t.r);
            d_tensors.push_back(t.d);
            h_tensors.push_back(t.h);
        }
        //The real part is straightforward
        auto r = torch::einsum(arg, r_tensors);

        // Find the position of the '->' in the einsum string
        auto posa = arg.find("->");

        //Find the positions of the "," in the einsum string
        std::vector<std::string> lhsargs{};
        std::string rhsarg; //The right hand side of the einsum only one output
        size_t pos = arg.find(',');
        size_t posl = 0;
        while(pos != std::string::npos) 
        {
           lhsargs.push_back(arg.substr(posl+1, pos));
           posl = pos;
        }
        size_t pos2 = arg.find('->');
        rhsarg = arg.substr(pos2+2);
        std::vector<torch::Tensor> dr_tensors{};

        auto d = torch::zeros_like(tensors[0].d);
        for ( int i=0; i < tensors.size(); i++)
        {
            auto dpart = torch::zeros_like(tensors[0].d);
            auto dl = tensors[i].d;
            for ( int j=0; j < tensors.size(); i++)
            {
                if ( i==j) continue;
                auto darg = lhsargs[i] + "z," + lhsargs[j] + "->" + rhsarg + "z";
                dl = dl+torch::einsum(darg, {dpart, d_tensors[j]});
            }
            d = d + dl;
            dr_tensors.push_back(dl); //Keep this for the h calculation

        }

        torch::Tensor h = torch::zeros_like(tensors[0].h);
        for ( int i=0; i < tensors.size(); i++)
        {
            for ( int j=0; j < tensors.size(); i++)
            {
                if ( i==j) continue;
                auto harg = lhsargs[i] + "zw," + lhsargs[j] + "->" + rhsarg + "zw";
                h = h + torch::einsum(harg, {tensors[i].h, d_tensors[j]});
            }
            //Now for the d^2 terms
            auto d2 = torch::zeros_like(tensors[0].d);
            for ( int j=0; j < tensors.size(); i++)
            {
                if ( i==j) continue;
                auto d2arg = lhsargs[i] + "z," + lhsargs[j] + "w->" + rhsarg + "zw";
                h = h + torch::einsum(d2arg, {d_tensors[i], dr_tensors[j]});
            }
        }
        
        return TensorHyperDual(r, d, h);
    }



};





/**
 * The TensorMatDual class is meant to keep track of the sensitivities to the initial conditions
 * of a function. It is a generalization of the TensorDual class, which keeps track of the first
 * order derivatives of a function. The TensorMatDual class keeps track
 * */

class TensorMatDual {
public:
    torch::Tensor r;
    torch::Tensor d;
    torch::Dtype dtype_ = torch::kFloat64;
    torch::Device device_ = torch::kCPU;

public:


    // Constructor
    TensorMatDual(torch::Tensor r, torch::Tensor d) {
        assert (r.dim() ==3 && "In TensorMatDual, the real part must be a matrix");
        assert (d.dim() ==4 && "Dual part of TensorMatDual must have four dimensions");

        this->r = r;
        this->d = d;
        dtype_ = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }

    // Add constructor for a TensorMatDual from a TensorMatDual
    TensorMatDual(const TensorMatDual& other) {
        this->r = other.r.clone();
        this->d = other.d.clone();
        this->device_ = other.device_;
    }

    TensorMatDual to(torch::Device device) {
        this->r = this->r.to(device);
        this->d = this->d.to(device);
        this->device_ = device;
        return *this;
    }

    torch::Device device() const {
        return this->device_;
    }




    TensorMatDual() {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU); // You need to specify the correct data type here

        // Create zero tensors with the specified options
        torch::Tensor rl{torch::zeros({1, 1, 1}, options)};
        torch::Tensor dl{torch::zeros({1, 1, 1, 1}, options)};
        TensorMatDual(rl, dl);

    }

    TensorMatDual(const TensorDual& x, int dim =2) {
        auto r = x.r.unsqueeze(dim);
        auto d = x.d.unsqueeze(dim);
        TensorMatDual(r, d);
    }

    TensorMatDual complex() {
        torch::Tensor rc, dc;
        this->r.is_complex() ? rc = this->r : rc = torch::complex(this->r, torch::zeros_like(this->r)).to(this->device_);
        this->d.is_complex() ? dc = this->d : dc = torch::complex(this->d, torch::zeros_like(this->d)).to(this->device_);
        return TensorMatDual(std::move(rc), std::move(dc));
    }


    friend std::ostream& operator<<(std::ostream& os, const TensorMatDual& obj){
        os << "r: " << obj.r << std::endl;
        os << "d: " << obj.d << std::endl;
        return os;
    }


    // Constructor overload for scalar types
    template <typename S>
    TensorMatDual(S r, S d, int64_t dim = 1) {
        auto options = torch::TensorOptions().dtype(torch::kFloat64); // You need to specify the correct data type here

        this->r = torch::tensor({r}, options);
        if (this->r.dim() == 1) {
            this->r = this->r.unsqueeze(dim);
        }

        this->d = torch::tensor({d}, options);
        if (this->d.dim() == 1) {
            this->d = this->d.unsqueeze(dim);
        }

    }

    TensorDual squeeze(int dim )
    {
        auto r = this->r.squeeze(dim);
        auto d = this->d.squeeze(dim);
        return TensorDual(r, d);
    }

    TensorMatDual contiguous()
    {
        auto r = this->r.contiguous();
        auto d = this->d.contiguous();
        return TensorMatDual(r, d);
    }


 
    TensorMatDual eye() {
        auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});
        auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(2), this->d.size(3)}, this->d.options());
        return TensorMatDual(r, d);
    }



    TensorMatDual sum(int dim){
        auto r = this->r.sum(dim, true);
        auto d = this->d.sum(dim, true);
        return TensorMatDual(r, d);
    }

    TensorMatDual square() const {
        auto rsq = r.square(); // Compute the square of the real part
        auto d = 2 * r.unsqueeze(-1) * this->d;
        return TensorMatDual(rsq, d);
    }

    TensorMatDual sqrt() const {
        auto r = torch::sqrt(this->r); // Compute the square root of the real part
        auto rf = torch::where(torch::real(r) > 0, r, torch::zeros_like(r)); // Remove negative elements
        auto d = torch::einsum("mij, mijn->mijn", {0.5*rf.pow(-0.5), this->d});
        return TensorMatDual(r, d);
    }

    TensorMatDual normL2()
    {
       auto norm_r = torch::norm(this->r, 2, -1, true);
       auto norm_r_expanded =norm_r.expand_as(this->r);
       auto grad_r = this->r / norm_r_expanded;
       auto dual = torch::einsum("mij, mijn->min", {grad_r, this->d}).unsqueeze(2);
       return TensorMatDual(norm_r, dual);
    }

    static TensorMatDual createZero(const torch::Tensor& r, int ddim) {
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(ddim); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        return TensorMatDual(r, ds);
    }

    TensorMatDual zeros_like(const torch::Tensor &x) const {
        auto rc = torch::zeros_like(x);
        int nr1 = d.size(1);
        int nr2 = d.size(2);
        int nd = d.size(3);
        auto dc = torch::zeros({nr1, nr2, nd}, x.dtype());
        if (r.dtype() == torch::kBool) {
            dc = torch::zeros({nr1, nr2, nd}, torch::kFloat64);
        }
        
        return TensorMatDual(rc, dc);
    }

    TensorMatDual zeros_like() {
        auto rc = torch::zeros_like(this->r);
        auto dc = torch::zeros_like(this->d);
        return TensorMatDual(rc, dc);
    }


    TensorMatDual clone() const {
        return TensorMatDual(this->r.clone(), this->d.clone());
    }

    TensorDual squeeze() {
        if (this->r.size(2) == 1) {
            return TensorDual(this->r.squeeze(2), this->d.squeeze(2));
        }
        auto r = this->r.squeeze(1);
        auto d = this->d.squeeze(1);
        return TensorDual(r, d);
    }
    
    
    /**
     * Defaults to dimension 2 for concatenation
     */
    static TensorMatDual cat(const TensorMatDual& t1, const TensorMatDual &t2)
    {
        auto r = torch::cat({t1.r, t2.r}, 2);
        auto d = torch::cat({t1.d, t2.d}, 2);
        return TensorMatDual(r, d);
    }

    static TensorMatDual cat(const TensorMatDual& t1, const TensorMatDual &t2, int dim)
    {
        auto r = torch::cat({t1.r, t2.r}, dim);
        auto d = torch::cat({t1.d, t2.d}, dim);
        return TensorMatDual(r, d);
    }

    static TensorMatDual cat(const TensorMatDual& t1, const TensorDual &t2)
    {
        auto r = torch::cat({t1.r, t2.r.unsqueeze(2)}, 2);
        auto d = torch::cat({t1.d, t2.d.unsqueeze(2)}, 2);
        return TensorMatDual(r, d);
    }

    static TensorMatDual cat(const TensorMatDual& t1, const torch::Tensor &t2)
    {
        auto rt = t2.repeat({t1.r.size(0), 1, 1});
        auto r = torch::cat({t1.r, rt}, 2);
        auto d = torch::cat({t1.d, t1.d*0}, 2);
        return TensorMatDual(r, d);
    }


    //overload the + operator
    TensorMatDual operator+(const TensorMatDual& other) const {
        return TensorMatDual(this->r + other.r, this->d + other.d);
    }

    //overload the + operator
    TensorMatDual operator+(const TensorDual& other) const {
        return TensorMatDual(this->r + other.r, this->d + other.d);
    }

    //overload the + operator for a double
    TensorMatDual operator+(const double& other) const {
        return TensorMatDual(this->r + other, this->d);
    }


    //overload the - operator
    TensorMatDual operator-(const TensorMatDual& other) const {
        return TensorMatDual(this->r - other.r, this->d - other.d);
    }

    //overload the - operator
    TensorMatDual operator-(const double& other) const {
        return TensorMatDual(this->r - other, this->d);
    }



    // Overload the equals operator for TensorDual == TensorDual
    torch::Tensor operator==(const TensorMatDual& other) const {
        auto mask = r == other.r;
        return torch::squeeze(mask, 2);
    }




    //overload the - operator
    TensorMatDual operator-() const {
        return TensorMatDual(-this->r, -this->d);
    }

    TensorMatDual operator*(const double other) const {
        auto real = this->r*other;
        auto dual = this->d*other;
        return TensorMatDual(real, dual);
    }



    
    TensorMatDual operator/(const TensorMatDual& other) const {
        auto r = this->r / other.r;
        auto otherrsq = other.r.square();
        auto d = -(this->r / otherrsq).unsqueeze(-1) * other.d + this->d/other.r.unsqueeze(-1);
        //std::cerr << "d sizes in /: " << d.sizes() << std::endl;
        //std::cerr << "current dual sizes " << this->d_.sizes() << std::endl;
        //Make sure the dimensions stay the same as the input tensor
        return TensorMatDual(r, d);
    }

    TensorMatDual operator/(const TensorDual& other) const {
        TensorMatDual other_mat = TensorMatDual::unsqueeze(other, 2);
        return (*this) / other_mat;
    }

    TensorMatDual operator/(const torch::Tensor& other) const {
        auto real = this->r/other;
        auto dual = this->d/other.unsqueeze(-1);
        return TensorMatDual(real, dual);
    }


    TensorMatDual operator/(const double other) const {
        auto real = this->r/other;
        auto dual = this->d/other;
        return TensorMatDual(real, dual);
    }

    TensorMatDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {
        auto r = this->r.index(indices);
        //Add a column if it is missing
        r.dim() == 2 ? r = r.unsqueeze(2) : r;
        auto d = this->d.index(indices);
        d.dim() == 3 ? d = d.unsqueeze(2) : d;
        return TensorMatDual(r, d);
    }


    TensorMatDual index(int index) {
        auto real = this->r.index({index});
        auto dual = this->d.index({index});
        return TensorMatDual(real, dual);
    }


    TensorMatDual index(const torch::Tensor& mask) {
        auto real = r.index({mask});
        auto dual = d.index({mask});
        return TensorMatDual(real, dual);
    }


    TensorMatDual index(const std::vector<TensorIndex>& index) {
        auto real = r.index(index);
        auto dual = d.index(index);
        return TensorMatDual(real, dual);
    }

    void requires_grad_(bool req_grad) {
        r.requires_grad_(req_grad);
        d.requires_grad_(req_grad);
    }

    void backward() {
        r.backward();
        d.backward();
    }

 


    TensorMatDual abs() const {
        auto abs_r = torch::abs(r); // Compute the absolute value of the real part
        auto sign_r = torch::is_complex(r) ? torch::sign(torch::real(r)) : torch::sign(r); // Compute the sign of the real part
        auto abs_d = sign_r.unsqueeze(-1) * d; // The dual part multiplies by the sign of the real part
        return TensorMatDual(abs_r, abs_d);
    }





    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     */
    static TensorDual einsum(const std::string& arg, 
                             const TensorMatDual& first, 
                             const TensorDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        auto d1 = torch::einsum(darg1, {first.d,  second.r});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        return TensorDual(std::move(r), std::move(d1 + d2));
    }

    static TensorHyperDual einsum(const std::string& arg, 
                                  const TensorMatHyperDual& first, 
                                  const TensorHyperDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        auto d1 = torch::einsum(darg1, {first.d,  second.r});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        auto d1d1r2arg = arg1 + "z," + arg1 + "w," + arg2+"->" + arg3 + "zw";
        auto d1d1r2 = torch::einsum(d1d1r2arg, {first.d, first.d, second.r});
        auto r1h1r2arg = arg1 + "," + arg1 + "zw," + arg2+"->" + arg3 + "zw";
        auto r1h1r2 = torch::einsum(r1h1r2arg, {first.r, first.h, second.r});
        auto r1d1d2arg = arg1 + "," + arg1 + "z," + arg2+"w->" + arg3 + "zw";
        auto r1d1d2 = torch::einsum(r1d1d2arg, {first.r, first.d, second.d});
        auto d1r2d2arg = arg1 + "z," + arg2 + "," + arg2+"w->" + arg3 + "zw";
        auto d1r2d2 = torch::einsum(d1r2d2arg, {first.d, second.r, second.d});
        auto r1d2d2arg = arg1 + "," + arg2 + "z," + arg2+"w->" + arg3 + "zw";
        auto r1d2d2 = torch::einsum(r1d2d2arg, {first.r, second.d, second.d});
        auto r1r2h2arg = arg1 + "," + arg2 + "," + arg2+"zw->" + arg3 + "zw";
        auto r1r2h2 = torch::einsum(r1r2h2arg, {first.r, second.r, second.h});

        return TensorHyperDual(std::move(r), 
                               std::move(d1 + d2), 
                               std::move(d1d1r2 + r1h1r2 + r1d1d2 + d1r2d2 + r1d2d2 + r1r2h2));
    }


    static TensorMatDual einsum(const std::string& arg, const TensorMatDual& first, const torch::Tensor& second) {

        auto r = torch::einsum(arg, {first.r, second});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1 = torch::einsum(darg1, {first.d,  second});

        return TensorMatDual(std::move(r), std::move(d1));
    }

    static TensorMatHyperDual einsum(const std::string& arg,
                                     const TensorMatHyperDual& first,
                                     const torch::Tensor& second) {
        auto r = torch::einsum(arg, {first.r, second});
        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d = torch::einsum(darg1, {first.d,  second});
        auto d1d1r2arg = arg1 + "z," + arg1 + "w," + arg2+"->" + arg3 + "zw";
        auto d1d1r2 = torch::einsum(d1d1r2arg, {first.d, first.d, second});
        auto r1h1r2arg = arg1 + "," + arg1 + "zw," + arg2+"->" + arg3 + "zw";
        auto r1h1r2 = torch::einsum(r1h1r2arg, {first.r, first.h, second});
        return TensorMatHyperDual(std::move(r), std::move(d), std::move(d1d1r2 + r1h1r2));
    }


    static TensorMatDual einsum(const std::string& arg, const TensorDual& first, const TensorMatDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        auto d1 = torch::einsum(darg1, {first.d,  second.r});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        return TensorMatDual(std::move(r), std::move(d1 + d2));
    }

    static TensorMatDual einsum(const std::string& arg, const torch::Tensor& first, const TensorMatDual& second) {

        auto r = torch::einsum(arg, {first, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "," + arg2 + "z->" + arg3 + "z";

        auto d = torch::einsum(darg1, {first,  second.d});

        return TensorMatDual(std::move(r), std::move(d));
    }



    static TensorMatDual einsum(const std::string& arg, const TensorMatDual& first, const TensorMatDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
 
        auto d1 = torch::einsum(darg1, {first.d,  second.r});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        return TensorMatDual(std::move(r), std::move(d1 + d2));
    }


    TensorMatDual max(int dim=1) {
         // max_values, max_indices = torch.max(self.r, dim=1, keepdim=True)
         auto max_result = torch::is_complex(r) ? torch::max(torch::real(this->r), /*dim=*/dim, /*keepdim=*/true) : 
                           torch::max(this->r, /*dim=*/dim, /*keepdim=*/true);
         auto max_indices = std::get<1>(max_result); // For the indices of the maximum values
         //auto max_values = std::get<0>(max_result); // For the maximum values

        //dshape = max_indices.unsqueeze(-1).expand(-1, -1, self.d.shape[-1])
        auto d_indices = max_indices.unsqueeze(-1).expand({-1, -1, -1, this->d.size(-1)});
        //dual_values = torch.gather(self.d, 1, dshape)
        auto real_values = torch::gather(this->r, dim, max_indices);
        auto dual_values = torch::gather(this->d, dim, d_indices);
        //return TensorDual(max_values, dual_values)
        return TensorMatDual(real_values, dual_values);
    }

    TensorMatDual min(int dim=1) {
         // max_values, max_indices = torch.max(self.r, dim=1, keepdim=True)
         auto min_result = torch::is_complex(r) ? torch::min(torch::real(this->r), /*dim=*/dim, /*keepdim=*/true) : 
                           torch::min(this->r, /*dim=*/dim, /*keepdim=*/true);
         auto min_indices = std::get<1>(min_result); // For the indices of the maximum values
         //auto max_values = std::get<0>(max_result); // For the maximum values

        //dshape = max_indices.unsqueeze(-1).expand(-1, -1, self.d.shape[-1])
        auto d_indices = min_indices.unsqueeze(-1).expand({-1, -1, -1, this->d.size(-1)});
        //dual_values = torch.gather(self.d, 1, dshape)
        auto real_values = torch::gather(this->r, dim, min_indices);
        auto dual_values = torch::gather(this->d, dim, d_indices);
        //return TensorDual(max_values, dual_values)
        return TensorMatDual(real_values, dual_values);
    }


    


    void index_put_(const torch::Tensor& mask, const TensorDual& value) {
        this->r.index_put_({mask}, value.r.squeeze());
        this->d.index_put_({mask}, value.d.squeeze());
    }


    void index_put_(const TensorIndex& mask, const TensorDual& value) {
        this->r.index_put_({mask}, value.r.squeeze());
        this->d.index_put_({mask}, value.d.squeeze());
    }


    void index_put_(const std::vector<TensorIndex>& mask, const TensorDual& value) {
        this->r.index_put_(mask, value.r);
        this->d.index_put_(mask, value.d);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const TensorMatDual& value) {
        this->r.index_put_(mask, value.r);
        this->d.index_put_(mask, value.d);  
    }

    

    void index_put_(const std::vector<TensorIndex>& mask, const double& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const torch::Tensor& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
    }


    static TensorMatDual unsqueeze(const TensorDual& x, int dim) {
        auto r = x.r.unsqueeze(dim);
        auto d = x.d.unsqueeze(dim);
        return TensorMatDual(r, d);
    }
};


/**
 * The TensorMatHyperDual class is meant to keep track of the second order sensitivities to the initial conditions
 * of a set of functions. It is a generalization of the TensorHyperDual class, which keeps track of the first
 * order derivatives of a function.
 * */
class TensorMatHyperDual {
public:
    torch::Tensor r;       // Real part [M, N, L]
    torch::Tensor d;       // Dual part [M, N, L, D]
    torch::Tensor h;       // Hyperdual part [M, N, L, H1, H2]
    torch::Dtype dtype_;   // Data type (default: Float64)
    torch::Device device_; // Device (default: CPU)
    
    // Constructor with tensors
    TensorMatHyperDual(torch::Tensor r, torch::Tensor d, torch::Tensor h) 
        : r(r), d(d), h(h), 
          dtype_(torch::typeMetaToScalarType(r.dtype())),
          device_(r.device()) {
        if (r.dim() != 3) {
            throw std::invalid_argument("Real part (r) must have dimensions [M, N, L].");
        }
        if (d.dim() != 4) {
            throw std::invalid_argument("Dual part (d) must have dimensions [M, N, L, D].");
        }
        if (h.dim() != 5) {
            throw std::invalid_argument("Hyperdual part (h) must have dimensions [M, N, L, H1, H2].");
        }
        if (r.size(0) != d.size(0) || r.size(0) != h.size(0) ||
            r.size(1) != d.size(1) || r.size(1) != h.size(1) ||
            r.size(2) != d.size(2) || r.size(2) != h.size(2)) {
            throw std::invalid_argument("Shape mismatch: r, d, and h must share the same [M, N, L] dimensions.");
        }
    }

    // Helper function to convert tensor sizes to a string
    static std::string sizes_to_string(const torch::Tensor& tensor) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < tensor.sizes().size(); ++i) {
            oss << tensor.sizes()[i];
            if (i != tensor.sizes().size() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    }

    /**
     * Constructor for TensorMatHyperDual.
     *
     * @param r The real part tensor [M, N, L].
     * @param d The dual part tensor [M, N, L, D].
     * @param h The hyperdual part tensor [M, N, L, D, D].
     * @throws std::invalid_argument if dimensions or types are invalid.
     */


    /**
     * Constructor for TensorMatHyperDual.
     */
    TensorMatHyperDual(torch::Tensor r, torch::Tensor d, torch::Tensor h) 
        : r(r), d(d), h(h), 
        dtype_(torch::typeMetaToScalarType(r.dtype())),
        device_(r.device()) {
        if (r.dim() != 3) {
            throw std::invalid_argument("Real part (r) must have dimensions [M, N, L], but got: " + 
                                        sizes_to_string(r));
        }
        if (d.dim() != 4) {
            throw std::invalid_argument("Dual part (d) must have dimensions [M, N, L, D], but got: " + 
                                        sizes_to_string(d));
        }
        if (h.dim() != 5) {
            throw std::invalid_argument("Hyperdual part (h) must have dimensions [M, N, L, D, D], but got: " + 
                                        sizes_to_string(h));
        }
        if (r.sizes().slice(0, 3) != d.sizes().slice(0, 3) || 
            r.sizes().slice(0, 3) != h.sizes().slice(0, 3)) {
            throw std::invalid_argument("Shape mismatch: r, d, and h must share the same [M, N, L] dimensions. "
                                        "Got r: " + sizes_to_string(r) + ", d: " + sizes_to_string(d) + 
                                        ", h: " + sizes_to_string(h));
        }
        if (h.size(3) != d.size(3) || h.size(4) != d.size(3)) {
            throw std::invalid_argument("Hyperdual dimensions [D, D] must match the dual dimensions [D]. "
                                        "Got d: " + sizes_to_string(d) + ", h: " + sizes_to_string(h));
        }
    }
    /**
     * Create a new TensorMatHyperDual object with tensors moved to the specified device.
     *
     * @param device The device to move the tensors to (e.g., torch::kCUDA, torch::kCPU).
     * @return A new TensorMatHyperDual object with all components moved to the specified device.
     */
    TensorMatHyperDual to(torch::Device device) const {
        return TensorMatHyperDual(
            this->r.to(device),  // Move the real part
            this->d.to(device),  // Move the dual part
            this->h.to(device)   // Move the hyperdual part
        );
    }

    /**
     * Get the device of the TensorMatHyperDual object.
     *
     * @return The torch::Device where the tensors are stored.
     */
    torch::Device device() const {
        return this->device_;
    }

    TensorMatHyperDual(const TensorDual& x, int dim = 2)
        : r(x.r.unsqueeze(dim)), 
        d(x.d.unsqueeze(dim)), 
        h(torch::zeros_like(x.d.unsqueeze(dim))),
        dtype_(torch::typeMetaToScalarType(x.r.dtype())), // Explicit conversion
        device_(x.r.device()) {}

    /**
     * Constructor to create a TensorMatHyperDual object from a TensorDual object.
     *
     * @param x The TensorDual object to extend into a TensorMatHyperDual.
     * @param dim The dimension along which to unsqueeze the tensors (default: 2).
     * @throws std::invalid_argument if the specified dimension is invalid.
     */
    TensorMatHyperDual(const TensorDual& x, int dim = 2)
        : r(x.r.unsqueeze(dim)), 
        d(x.d.unsqueeze(dim)), 
        h(torch::zeros_like(x.d.unsqueeze(dim))), // Initialize hyperdual part with zeros
        dtype_(torch::typeMetaToScalarType(x.r.dtype())), 
        device_(x.r.device()) {
        if (dim < 0 || dim > x.r.dim()) {
            throw std::invalid_argument("Invalid dimension for unsqueeze: " + std::to_string(dim) +
                                        ". Must be in the range [0, " + std::to_string(x.r.dim()) + "].");
        }
    }    
    
    TensorMatHyperDual complex() const {
        torch::Tensor rc, dc, hc;

        // Convert real tensors to complex tensors
        if (this->r.is_complex()) {
            rc = this->r;
        } else {
            rc = torch::cat({this->r.unsqueeze(-1), torch::zeros_like(this->r).unsqueeze(-1)}, -1);
        }

        if (this->d.is_complex()) {
            dc = this->d;
        } else {
            dc = torch::cat({this->d.unsqueeze(-1), torch::zeros_like(this->d).unsqueeze(-1)}, -1);
        }

        if (this->h.is_complex()) {
            hc = this->h;
        } else {
            hc = torch::cat({this->h.unsqueeze(-1), torch::zeros_like(this->h).unsqueeze(-1)}, -1);
        }

        // Return a new TensorMatHyperDual with complex tensors
        return TensorMatHyperDual(rc, dc, hc);
    }


    /**
     * Convert the TensorMatHyperDual object to use complex tensors.
     *
     * If the tensors are already complex, they are returned as is. Otherwise,
     * real tensors are converted to complex tensors by appending a zero imaginary part.
     *
     * @return A new TensorMatHyperDual object with complex tensors.
     */
    TensorMatHyperDual complex() const {
        // Convert real part to complex
        auto rc = this->r.is_complex()
                    ? this->r
                    : torch::cat({this->r.unsqueeze(-1), torch::zeros_like(this->r).unsqueeze(-1)}, -1);

        // Convert dual part to complex
        auto dc = this->d.is_complex()
                    ? this->d
                    : torch::cat({this->d.unsqueeze(-1), torch::zeros_like(this->d).unsqueeze(-1)}, -1);

        // Convert hyperdual part to complex
        auto hc = this->h.is_complex()
                    ? this->h
                    : torch::cat({this->h.unsqueeze(-1), torch::zeros_like(this->h).unsqueeze(-1)}, -1);

        // Return a new TensorMatHyperDual with complex tensors
        return TensorMatHyperDual(rc, dc, hc);
    }
    /**
     * Extract the real part of the TensorMatHyperDual object.
     *
     * If the tensors are already real, they are returned as is.
     *
     * @return A new TensorMatHyperDual object with the real parts of the tensors.
     */
    TensorMatHyperDual real() const {
        // Extract the real part of each tensor
        auto r = this->r.is_complex() ? torch::real(this->r) : this->r;
        auto d = this->d.is_complex() ? torch::real(this->d) : this->d;
        auto h = this->h.is_complex() ? torch::real(this->h) : this->h;

        // Return a new TensorMatHyperDual with real parts
        return TensorMatHyperDual(r, d, h);
    }

    /**
     * Extract the imaginary part of the TensorMatHyperDual object.
     *
     * If the tensors are real, the imaginary part is zero tensors of the same shape.
     *
     * @return A new TensorMatHyperDual object with the imaginary parts of the tensors.
     */
    TensorMatHyperDual imag() const {
        // Extract the imaginary part or return zero tensors if already real
        auto r = this->r.is_complex() ? torch::imag(this->r) : torch::zeros_like(this->r);
        auto d = this->d.is_complex() ? torch::imag(this->d) : torch::zeros_like(this->d);
        auto h = this->h.is_complex() ? torch::imag(this->h) : torch::zeros_like(this->h);

        // Return a new TensorMatHyperDual with imaginary parts
        return TensorMatHyperDual(r, d, h);
    }

    /**
     * Compute the absolute value of the TensorMatHyperDual object.
     *
     * For the absolute value:
     * - The real part becomes |r|.
     * - The dual part is scaled by the sign of the real part.
     * - The hyperdual part is zero because the second derivative of |r| is undefined at r = 0.
     *
     * @return A new TensorMatHyperDual object with the absolute value applied.
     */
    TensorMatHyperDual abs() const {
        // Compute the absolute value of the real part
        auto abs_r = torch::abs(this->r);

        // Compute the sign of the real part
        auto sign_r = torch::sign(this->r);

        // Scale the dual part by the sign of the real part
        auto abs_d = sign_r.unsqueeze(-1) * this->d;

        // Hyperdual part is zero
        auto abs_h = torch::zeros_like(this->h);

        return TensorMatHyperDual(abs_r, abs_d, abs_h);
    }


    /**
     * Compute the maximum values along a specified dimension for the TensorMatHyperDual object.
     *
     * The real, dual, and hyperdual parts are reduced based on the indices of the maximum values.
     *
     * @param dim The dimension along which to compute the maximum (default: 1).
     * @return A new TensorMatHyperDual object containing the maximum values and corresponding dual and hyperdual values.
     */
    TensorMatHyperDual max(int dim = 1) const {
        // Compute max values and indices along the specified dimension
        auto max_result = torch::max(this->r, dim, /*keepdim=*/true);
        auto max_values = std::get<0>(max_result);  // Maximum values
        auto max_indices = std::get<1>(max_result); // Indices of the maximum values

        // Adjust the shape of max_indices for gathering
        auto dshape = max_indices.unsqueeze(-1).expand_as(this->d.select(dim, 0)); // Expand to match dual shape
        auto hshape = max_indices.unsqueeze(-1).unsqueeze(-1).expand_as(this->h.select(dim, 0).select(dim, 0)); // Expand to match hyperdual shape

        // Gather dual and hyperdual values based on max indices
        auto dual_values = torch::gather(this->d, dim, dshape);
        auto hyper_values = torch::gather(this->h, dim, hshape);

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(max_values, dual_values, hyper_values);
    }

    /**
     * Compute the minimum values along a specified dimension for the TensorMatHyperDual object.
     *
     * The real, dual, and hyperdual parts are reduced based on the indices of the minimum values.
     *
     * @param dim The dimension along which to compute the minimum (default: 1).
     * @return A new TensorMatHyperDual object containing the minimum values and corresponding dual and hyperdual values.
     */
    TensorMatHyperDual min(int dim = 1) const {
        // Compute min values and indices along the specified dimension
        auto min_result = torch::min(this->r, dim, /*keepdim=*/true);
        auto min_values = std::get<0>(min_result);  // Minimum values
        auto min_indices = std::get<1>(min_result); // Indices of the minimum values

        // Adjust the shape of min_indices for gathering
        auto dshape = min_indices.unsqueeze(-1).expand_as(this->d.select(dim, 0)); // Expand to match dual shape
        auto hshape = min_indices.unsqueeze(-1).unsqueeze(-1).expand_as(this->h.select(dim, 0).select(dim, 0)); // Expand to match hyperdual shape

        // Gather dual and hyperdual values based on min indices
        auto dual_values = torch::gather(this->d, dim, dshape);
        auto hyper_values = torch::gather(this->h, dim, hshape);

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(min_values, dual_values, hyper_values);
    }

    /**
     * Compute the sum of the TensorMatHyperDual object along a specified dimension.
     *
     * This method sums the real, dual, and hyperdual parts along the specified dimension, keeping the dimension.
     *
     * @param dim The dimension along which to compute the sum.
     * @return A new TensorMatHyperDual object containing the summed values.
     */
    TensorMatHyperDual sum(int dim) const {
        // Compute the sum for each part
        auto r_sum = this->r.sum(dim, /*keepdim=*/true);
        auto d_sum = this->d.sum(dim, /*keepdim=*/true);
        auto h_sum = this->h.sum(dim, /*keepdim=*/true);

        // Return a new TensorMatHyperDual object with summed components
        return TensorMatHyperDual(r_sum, d_sum, h_sum);
    }


    /**
     * Compute the square of the TensorMatHyperDual object.
     *
     * This method computes:
     * - Real part: \( r^2 \)
     * - Dual part: \( 2r \cdot d \)
     * - Hyperdual part: \( 2(d \cdot d + r \cdot h) \)
     *
     * @return A new TensorMatHyperDual object representing the square of the input.
     */
    TensorMatHyperDual square() const {
        // Compute the square of the real part
        auto rsq = r.square();

        // Compute the dual part: 2 * r * d
        auto dn = 2 * r.unsqueeze(-1) * this->d;

        // Compute the hyperdual part: 2 * (d * d + r * h)
        auto hn = 2 * torch::einsum("mij, mik->mijk", {d, d}) + 
                2 * torch::einsum("mi, mijk->mijk", {r, h});

        // Return the squared TensorMatHyperDual
        return TensorMatHyperDual(rsq, dn, hn);
    }



    /**
     * Overload the stream insertion operator for TensorMatHyperDual.
     *
     * This method provides a formatted string representation of the TensorMatHyperDual object, 
     * displaying the real, dual, and hyperdual parts.
     *
     * @param os The output stream.
     * @param obj The TensorMatHyperDual object to be printed.
     * @return The output stream with the object data appended.
     */
    friend std::ostream& operator<<(std::ostream& os, const TensorMatHyperDual& obj) {
        os << "TensorMatHyperDual {" << std::endl;
        os << "  r: " << obj.r << std::endl;
        os << "  d: " << obj.d << std::endl;
        os << "  h: " << obj.h << std::endl;
        os << "}";
        return os;
    }


    /**
     * Constructor overload for scalar types to create a TensorMatHyperDual.
     *
     * This constructor initializes the real, dual, and hyperdual parts from scalar values,
     * and adjusts dimensions to align with the TensorMatHyperDual structure.
     *
     * @param r Scalar value for the real part.
     * @param d Scalar value for the dual part.
     * @param h Scalar value for the hyperdual part.
     * @param dim The dimension along which to unsqueeze tensors (default: 1).
     */
    template <typename S>
    TensorMatHyperDual(S r, S d, S h, int64_t dim = 1) {
        // Tensor options for double precision
        auto options = torch::TensorOptions().dtype(torch::kFloat64);

        // Initialize the real part
        this->r = torch::tensor(r, options).unsqueeze(dim);

        // Initialize the dual part
        this->d = torch::tensor(d, options).unsqueeze(dim).unsqueeze(-1);

        // Initialize the hyperdual part
        this->h = torch::tensor(h, options).unsqueeze(dim).unsqueeze(-1).unsqueeze(-1);
    }



    
    //Forward declaration for eye function
    TensorMatHyperDual eye();
    /**
     * Squeeze the specified dimension of the TensorHyperDual object.
     *
     * Removes the specified dimension if it has size 1 from the real, dual, and hyperdual tensors.
     *
     * @param dim The dimension to squeeze. If the dimension does not have size 1, it remains unchanged.
     * @return A new TensorHyperDual object with the specified dimension squeezed.
     */
    TensorHyperDual squeeze(int dim) const {
        // Squeeze the specified dimension for each part
        auto r_squeezed = this->r.squeeze(dim);
        auto d_squeezed = this->d.squeeze(dim);
        auto h_squeezed = this->h.squeeze(dim);

        // Return the squeezed TensorHyperDual
        return TensorHyperDual(r_squeezed, d_squeezed, h_squeezed);
    }

    /**
     * Ensure that all components of the TensorMatHyperDual object are stored in contiguous memory.
     *
     * If the tensors are already contiguous, this method has no effect. Otherwise, it makes a contiguous copy.
     *
     * @return A new TensorMatHyperDual object with all tensors stored contiguously.
     */
    TensorMatHyperDual contiguous() const {
        // Ensure contiguous storage for each part
        auto r_contiguous = this->r.contiguous();
        auto d_contiguous = this->d.contiguous();
        auto h_contiguous = this->h.contiguous();

        // Return a new TensorMatHyperDual with contiguous tensors
        return TensorMatHyperDual(r_contiguous, d_contiguous, h_contiguous);
    }

    /**
     * Compute the square root of the TensorMatHyperDual object.
     *
     * For the square root operation:
     * - Real part: \( r_{\text{sqrt}} = \sqrt{r} \)
     * - Dual part: \( d_{\text{sqrt}} = \frac{1}{2 \sqrt{r}} \cdot d \)
     * - Hyperdual part: \( h_{\text{sqrt}} = \frac{1}{2 \sqrt{r}} \cdot h - \frac{1}{4 r^{3/2}} \cdot d \cdot d \)
     *
     * @return A new TensorMatHyperDual object representing the square root.
     * @throws std::invalid_argument if any element of `r` is negative.
     */
    TensorMatHyperDual sqrt() const {
        // Compute the square root of the real part
        if ((this->r < 0).any().item<bool>()) {
            throw std::invalid_argument("Square root of negative elements is not supported.");
        }
        auto r_sqrt = torch::sqrt(this->r);

        // Compute the dual part: d / (2 * sqrt(r))
        auto rf_inv_sqrt = 0.5 * r_sqrt.pow(-1); // 1 / (2 * sqrt(r))
        auto d_sqrt = torch::einsum("mij, mijn->mijn", {rf_inv_sqrt, this->d});

        // Compute the hyperdual part: h / (2 * sqrt(r)) - d * d / (4 * r^(3/2))
        auto rf_inv_3_sqrt = 0.25 * r_sqrt.pow(-3); // 1 / (4 * r^(3/2))
        auto h_sqrt = torch::einsum("mij, mijkn->mijkn", {rf_inv_sqrt, this->h}) -
                    torch::einsum("mij, mijk, mij->mijkn", {rf_inv_3_sqrt, this->d, this->d});

        // Return the new TensorMatHyperDual object
        return TensorMatHyperDual(r_sqrt, d_sqrt, h_sqrt);
    }

    /**
     * Compute the L2 norm of the TensorMatHyperDual object along the last dimension.
     *
     * For the L2 norm operation:
     * - Real part: \( ||r||_2 \)
     * - Dual part: \( \frac{r}{||r||_2} \cdot d \)
     * - Hyperdual part: Derived from the chain rule, involving second derivatives of the norm.
     *
     * @return A new TensorMatHyperDual object representing the L2 norm and its derivatives.
     */
    TensorMatHyperDual normL2() const {
        // Compute the L2 norm of the real part along the last dimension
        auto norm_r = torch::norm(this->r, 2, /*dim=*/-1, /*keepdim=*/true);

        // Avoid division by zero: Replace zeros in norm_r with a small epsilon
        auto norm_r_safe = torch::where(norm_r > 0, norm_r, torch::ones_like(norm_r) * 1e-12);

        // Gradient of the norm w.r.t. r: r / ||r||_2
        auto grad_r = this->r / norm_r_safe.expand_as(this->r);

        // Dual part: grad_r * d
        auto dual = torch::einsum("mij, mijn->min", {grad_r, this->d}).unsqueeze(2);

        // Compute the gradient of grad_r (second derivative of the norm w.r.t. r)
        auto grad_grad_r = torch::eye(this->r.size(-1), this->r.options())
                            .unsqueeze(0)
                            .expand({this->r.size(0), this->r.size(-1), this->r.size(-1)}) -
                        torch::einsum("mij,mik->mijk", {grad_r, grad_r});

        // Hyperdual part: grad_r * h - grad_grad_r * d
        auto hyperdual = torch::einsum("mij, mijnk->mink", {grad_r, this->h}) -
                        torch::einsum("mijk, mijn->mink", {grad_grad_r, this->d});

        // Return the TensorMatHyperDual with norm, dual, and hyperdual parts
        return TensorMatHyperDual(norm_r, dual, hyperdual);
    }

    /**
     * Create a TensorMatHyperDual object with zero-initialized dual and hyperdual parts.
     *
     * @param r The real part tensor [M, N, L].
     * @param ddim The dual dimension (number of sensitivities to track).
     * @return A new TensorMatHyperDual object with zero-initialized dual and hyperdual parts.
     */
    static TensorMatHyperDual createZero(const torch::Tensor& r, int ddim) {
        // Validate input dimensions
        if (r.dim() != 3) {
            throw std::invalid_argument("Real part tensor must have dimensions [M, N, L].");
        }

        // Create the shape for the dual tensor
        auto dshape = r.sizes().vec();  // Copy the sizes to a vector
        dshape.push_back(ddim);         // Add the dual dimension

        // Create the shape for the hyperdual tensor
        auto hshape = r.sizes().vec();  // Copy the sizes to a vector
        hshape.push_back(ddim);         // Add the first hyperdual dimension
        hshape.push_back(ddim);         // Add the second hyperdual dimension

        // Create zero tensors for the dual and hyperdual parts
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        auto hs = torch::zeros(hshape, options);

        // Return the TensorMatHyperDual object
        return TensorMatHyperDual(r, ds, hs);
    }


    /**
     * Create a TensorMatHyperDual object with zero-initialized components matching the dimensions of the input tensor `x`.
     *
     * This method generates:
     * - A real part tensor with zeros matching `x`.
     * - A dual part tensor with zeros matching the batch and spatial dimensions of `d` but with an additional dual dimension.
     * - A hyperdual part tensor with zeros matching the batch and spatial dimensions of `h` but with two additional hyperdual dimensions.
     *
     * @param x The input tensor whose shape and dtype determine the real part.
     * @return A TensorMatHyperDual object with zero-initialized components.
     */
    TensorMatHyperDual zeros_like(const torch::Tensor &x) const {
        // Create a real part tensor with zeros matching `x`
        auto rc = torch::zeros_like(x);

        // Retrieve dimensions for dual and hyperdual tensors
        auto nr1 = d.size(1); // Number of rows
        auto nr2 = d.size(2); // Number of columns
        auto nd = d.size(3);  // Dual dimension
        auto nh = h.size(4);  // Hyperdual dimension

        // Create dual part tensor
        auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
        auto dc = torch::zeros({x.size(0), nr1, nr2, nd}, options);

        // Create hyperdual part tensor
        auto hc = torch::zeros({x.size(0), nr1, nr2, nd, nh}, options);

        // Return the zero-initialized TensorMatHyperDual
        return TensorMatHyperDual(rc, dc, hc);
    }


    /**
     * Create a deep copy of the TensorMatHyperDual object.
     *
     * This method clones the real, dual, and hyperdual tensors to ensure that
     * the new object is independent of the original.
     *
     * @return A new TensorMatHyperDual object that is a deep copy of the current object.
     */
    TensorMatHyperDual clone() const {
        // Clone the real, dual, and hyperdual parts
        return TensorMatHyperDual(this->r.clone(), this->d.clone(), this->h.clone());
    }


    /**
     * Squeeze singleton dimensions (dimensions of size 1) from the TensorHyperDual object.
     *
     * If a specific dimension is of size 1, it will be removed. If no dimension is specified,
     * all singleton dimensions are squeezed.
     *
     * @return A new TensorHyperDual object with the specified dimensions squeezed.
     */
    TensorHyperDual squeeze(int dim = -1) const {
        // Check if a specific dimension is provided
        if (dim >= 0) {
            // Ensure the specified dimension can be squeezed
            if (this->r.size(dim) != 1) {
                throw std::invalid_argument("Specified dimension is not of size 1 and cannot be squeezed.");
            }
            // Squeeze the specified dimension
            return TensorHyperDual(this->r.squeeze(dim), this->d.squeeze(dim), this->h.squeeze(dim));
        }

        // If no dimension is specified, squeeze all singleton dimensions
        return TensorHyperDual(this->r.squeeze(), this->d.squeeze(), this->h.squeeze());
    }

    /**
     * Concatenate two TensorMatHyperDual objects along a specified dimension.
     *
     * This method concatenates the real, dual, and hyperdual parts of two
     * TensorMatHyperDual objects along the specified dimension.
     *
     * @param t1 The first TensorMatHyperDual object.
     * @param t2 The second TensorMatHyperDual object.
     * @param dim The dimension along which to concatenate (default: 2).
     * @return A new TensorMatHyperDual object with concatenated components.
     * @throws std::invalid_argument if the tensors cannot be concatenated due to mismatched dimensions.
     */
    static TensorMatHyperDual cat(const TensorMatHyperDual& t1, const TensorMatHyperDual& t2, int dim = 2) {
        // Validate that the tensors can be concatenated
        if (t1.r.sizes() != t2.r.sizes() && dim != 2) {
            throw std::invalid_argument("The shapes of t1 and t2 are incompatible for concatenation along dimension " + std::to_string(dim));
        }

        // Concatenate the real, dual, and hyperdual parts
        auto r = torch::cat({t1.r, t2.r}, dim);
        auto d = torch::cat({t1.d, t2.d}, dim);
        auto h = torch::cat({t1.h, t2.h}, dim);

        // Return the concatenated TensorMatHyperDual
        return TensorMatHyperDual(r, d, h);
    }

    /**
     * Concatenate two TensorMatHyperDual objects along a specified dimension.
     *
     * This method concatenates the real, dual, and hyperdual parts of two
     * TensorMatHyperDual objects along the specified dimension.
     *
     * @param t1 The first TensorMatHyperDual object.
     * @param t2 The second TensorMatHyperDual object.
     * @param dim The dimension along which to concatenate (default: 2).
     * @return A new TensorMatHyperDual object with concatenated components.
     * @throws std::invalid_argument if the tensors cannot be concatenated due to mismatched dimensions.
     */
    static TensorMatHyperDual cat(const TensorMatHyperDual& t1, const TensorMatHyperDual& t2, int dim = 2) {
        // Validate that the tensors can be concatenated
        if (t1.r.sizes() != t2.r.sizes() && dim != 2) {
            throw std::invalid_argument("The shapes of t1 and t2 are incompatible for concatenation along dimension " + std::to_string(dim));
        }

        // Concatenate the real, dual, and hyperdual parts
        auto r = torch::cat({t1.r, t2.r}, dim);
        auto d = torch::cat({t1.d, t2.d}, dim);
        auto h = torch::cat({t1.h, t2.h}, dim);

        // Return the concatenated TensorMatHyperDual
        return TensorMatHyperDual(r, d, h);
    }

    /**
     * Concatenate a TensorMatHyperDual object with a plain torch::Tensor along the third dimension.
     *
     * The plain tensor is broadcast and repeated along the batch dimension to match
     * the TensorMatHyperDual structure. The dual and hyperdual parts of the resulting
     * TensorMatHyperDual object are zero-initialized for the appended tensor.
     *
     * @param t1 The TensorMatHyperDual object.
     * @param t2 The torch::Tensor to concatenate.
     * @return A new TensorMatHyperDual object with concatenated components.
     * @throws std::invalid_argument if the dimensions of t2 are incompatible.
     */
    static TensorMatHyperDual cat(const TensorMatHyperDual& t1, const torch::Tensor& t2) {
        // Validate dimensions of t2
        if (t2.dim() != 2 || t2.size(0) != t1.r.size(1)) {
            throw std::invalid_argument(
                "The input tensor t2 must have dimensions [N, L], where N matches the row dimension of t1.");
        }

        // Repeat t2 along the batch dimension to match t1
        auto rt = t2.unsqueeze(0).expand({t1.r.size(0), t2.size(0), t2.size(1)});

        // Concatenate the real parts
        auto r = torch::cat({t1.r, rt}, 2);

        // Concatenate the dual parts with zero tensors for the added data
        auto zero_d = torch::zeros({t1.d.size(0), t1.d.size(1), t2.size(1), t1.d.size(3)}, t1.d.options());
        auto d = torch::cat({t1.d, zero_d}, 2);

        // Concatenate the hyperdual parts with zero tensors for the added data
        auto zero_h = torch::zeros({t1.h.size(0), t1.h.size(1), t2.size(1), t1.h.size(3), t1.h.size(4)}, t1.h.options());
        auto h = torch::cat({t1.h, zero_h}, 2);

        // Return the new TensorMatHyperDual
        return TensorMatHyperDual(r, d, h);
    }
    


    /**
     * Overload the + operator for TensorMatHyperDual objects.
     *
     * This method performs element-wise addition of the real, dual, and hyperdual parts
     * of two TensorMatHyperDual objects. Both objects must have compatible shapes.
     *
     * @param other The TensorMatHyperDual object to add.
     * @return A new TensorMatHyperDual object representing the element-wise sum.
     * @throws std::invalid_argument if the dimensions of the two objects are incompatible.
     */
    TensorMatHyperDual operator+(const TensorMatHyperDual& other) const {
        // Validate that the dimensions of the two objects are compatible
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes() || this->h.sizes() != other.h.sizes()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching dimensions for addition.");
        }

        // Perform element-wise addition for real, dual, and hyperdual parts
        auto r_sum = this->r + other.r;
        auto d_sum = this->d + other.d;
        auto h_sum = this->h + other.h;

        // Return the result as a new TensorMatHyperDual object
        return TensorMatHyperDual(r_sum, d_sum, h_sum);
    }
    /**
     * Overload the + operator for TensorMatHyperDual and TensorHyperDual objects.
     *
     * This operator performs element-wise addition of a TensorMatHyperDual object
     * with a TensorHyperDual object. The TensorHyperDual components are expanded
     * along dimension 1 to align with the TensorMatHyperDual structure.
     *
     * @param other The TensorHyperDual object to add.
     * @return A new TensorMatHyperDual object representing the element-wise sum.
     * @throws std::invalid_argument if the dimensions of the two objects are incompatible.
     */
    TensorMatHyperDual operator+(const TensorHyperDual& other) const {
        // Validate compatibility of dimensions
        if (this->r.size(0) != other.r.size(0) || this->r.size(2) != other.r.size(1)) {
            throw std::invalid_argument(
                "TensorHyperDual dimensions are incompatible with TensorMatHyperDual for addition.");
        }

        // Perform element-wise addition with broadcasting
        auto r_sum = this->r + other.r.unsqueeze(1);
        auto d_sum = this->d + other.d.unsqueeze(1);
        auto h_sum = this->h + other.h.unsqueeze(1);

        // Return the result as a new TensorMatHyperDual object
        return TensorMatHyperDual(r_sum, d_sum, h_sum);
    }
    /**
     * Overload the + operator for TensorMatHyperDual and a scalar (double).
     *
     * This operator performs element-wise addition of the scalar to the real part
     * of the TensorMatHyperDual object, leaving the dual and hyperdual parts unchanged.
     *
     * @param other The scalar (double) to add to the real part.
     * @return A new TensorMatHyperDual object with the scalar added to the real part.
     */
    TensorMatHyperDual operator+(const double& other) const {
        // Perform element-wise addition of the scalar to the real part
        auto r_sum = this->r + other;

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(r_sum, this->d, this->h);
    }


    /**
     * Overload the - operator for TensorMatHyperDual objects.
     *
     * This operator performs element-wise subtraction of the real, dual, and hyperdual parts
     * of one TensorMatHyperDual object from another. The dimensions of the two objects must match.
     *
     * @param other The TensorMatHyperDual object to subtract.
     * @return A new TensorMatHyperDual object representing the element-wise difference.
     * @throws std::invalid_argument if the dimensions of the two objects do not match.
     */
    TensorMatHyperDual operator-(const TensorMatHyperDual& other) const {
        // Validate compatibility of dimensions
        if (this->r.sizes() != other.r.sizes() ||
            this->d.sizes() != other.d.sizes() ||
            this->h.sizes() != other.h.sizes()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching dimensions for subtraction.");
        }

        // Perform element-wise subtraction
        auto r_diff = this->r - other.r;
        auto d_diff = this->d - other.d;
        auto h_diff = this->h - other.h;

        // Return the result as a new TensorMatHyperDual object
        return TensorMatHyperDual(r_diff, d_diff, h_diff);
    }

    /**
     * Overload the - operator for TensorMatHyperDual and a scalar (double).
     *
     * This operator subtracts the scalar from the real part of the TensorMatHyperDual object,
     * leaving the dual and hyperdual parts unchanged.
     *
     * @param other The scalar (double) to subtract from the real part.
     * @return A new TensorMatHyperDual object with the scalar subtracted from the real part.
     */
    TensorMatHyperDual operator-(const double& other) const {
        // Perform element-wise subtraction of the scalar from the real part
        auto r_diff = this->r - other;

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(r_diff, this->d, this->h);
    }

    /**
     * Overload the equality operator (==) for TensorMatHyperDual objects.
     *
     * This operator compares the real parts of two TensorMatHyperDual objects element-wise
     * and returns a tensor mask indicating where the real parts are equal. The dimensions
     * of the two objects must match for comparison.
     *
     * @param other The TensorMatHyperDual object to compare with.
     * @return A torch::Tensor mask indicating element-wise equality of the real parts.
     * @throws std::invalid_argument if the dimensions of the two objects do not match.
     */
    torch::Tensor operator==(const TensorMatHyperDual& other) const {
        // Validate compatibility of dimensions
        if (this->r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching dimensions for comparison.");
        }

        // Perform element-wise comparison of the real parts
        auto mask = this->r == other.r;

        // Optionally squeeze dimension 2 if necessary (assuming mask is 3D)
        if (mask.size(2) == 1) {
            return torch::squeeze(mask, 2);
        }

        return mask; // Return the mask as is if no squeezing is required
    }


    /**
     * Overload the unary - operator for TensorMatHyperDual objects.
     *
     * This operator negates the real, dual, and hyperdual parts of the TensorMatHyperDual object.
     *
     * @return A new TensorMatHyperDual object with all components negated.
     */
    TensorMatHyperDual operator-() const {
        // Negate the real, dual, and hyperdual parts
        return TensorMatHyperDual(-this->r, -this->d, -this->h);
    }

    /**
     * Overload the * operator for TensorMatHyperDual and a scalar (double).
     *
     * This operator scales the real, dual, and hyperdual parts of the TensorMatHyperDual
     * object by the given scalar.
     *
     * @param other The scalar (double) to multiply with.
     * @return A new TensorMatHyperDual object with all components scaled by the scalar.
     */
    TensorMatHyperDual operator*(const double other) const {
        // Scale each component by the scalar
        auto real = this->r * other;
        auto dual = this->d * other;
        auto hyper = this->h * other;

        // Return the scaled TensorMatHyperDual object
        return TensorMatHyperDual(real, dual, hyper);
    }

    /**
     * Overload the / operator for TensorMatHyperDual objects.
     *
     * This operator performs element-wise division of two TensorMatHyperDual objects.
     * It computes the division for the real, dual, and hyperdual parts while ensuring
     * correctness in propagating derivatives.
     *
     * @param other The TensorMatHyperDual object to divide by.
     * @return A new TensorMatHyperDual object representing the element-wise division.
     * @throws std::invalid_argument if the dimensions of the two objects do not match.
     */
    TensorMatHyperDual operator/(const TensorMatHyperDual& other) const {
        // Validate compatibility of dimensions
        if (this->r.sizes() != other.r.sizes() ||
            this->d.sizes() != other.d.sizes() ||
            this->h.sizes() != other.h.sizes()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching dimensions for division.");
        }

        // Extract components
        auto r1 = this->r;
        auto d1 = this->d;
        auto h1 = this->h;
        auto r2 = other.r;
        auto d2 = other.d;
        auto h2 = other.h;

        auto r2_safe = r2.clamp_min(1e-12); // Replace zeros or near-zeros with a small epsilon

        // Real part of the result
        auto rn = r1 / r2_safe;
        //Cache some values to avoid recomputation
        auto r2_inv = r2_safe.reciprocal(); // r2^(-1)
        auto r2_inv2 = r2_inv * r2_inv;     // r2^(-2)
        auto r2_inv3 = r2_inv2 * r2_inv;    // r2^(-3)
        

        // Dual part of the result
        //-r1/r2^2 * d2 + r2^(-1) * d1
        auto dn = torch::einsum("mij, mijk->mijk", {-r1*r2_inv2, d2}) +
                  torch::einsum("mij, mijk->mijk", {r2_inv, d1});

        auto d1d2 = torch::einsum("mijk, mijl->mijkl", {d1, d2});
        auto d2d2 = torch::einsum("mijl, mijk->mijkl", {d2, d2});
        
        
        // Hyperdual part of the result
        //-d1d2/r2^2 + 2r1/r2^3 * d2d2 + r1/r2^2 * h2 + r2^(-1) * h1 - r2^(-2) * d1d2
        auto hn = torch::einsum("mijkl, mij->mijkl", {-d1d2, r2_inv2})+
                  2*torch::einsum("", {r1*r2_inv3, d2d2})+
                  torch::einsum("mij, mijkl->mijkl", {r1*r2_inv2, h2})+
                  torch::einsum("mij, mijkl->mijkl", {r2_inv, h1})+
                  torch::einsum("mij, mijkl->mijkl", {-r2_inv2, d1d2});
        // Return the result as a new TensorMatHyperDual object
        return TensorMatHyperDual(rn, dn, hn);
    }


    /**
     * Overload the / operator for TensorMatHyperDual and TensorHyperDual objects.
     *
     * This operator divides a TensorMatHyperDual object by a TensorHyperDual object.
     * The TensorHyperDual components are expanded along the appropriate dimensions
     * to match the structure of the TensorMatHyperDual. The hyperdual part of the
     * TensorHyperDual is assumed to be zero.
     *
     * @param other The TensorHyperDual object to divide by.
     * @return A new TensorMatHyperDual object representing the element-wise division.
     * @throws std::invalid_argument if the dimensions of the objects are incompatible.
     */
    TensorMatHyperDual operator/(const TensorHyperDual& other) const {
        // Expand TensorHyperDual components to match TensorMatHyperDual dimensions
        auto r2 = other.r.unsqueeze(1);                  // Add singleton dimension
        auto d2 = other.d.unsqueeze(1);                  // Add singleton dimension
        auto h2 = torch::zeros_like(this->h);            // Hyperdual part is zero

        // Construct a TensorMatHyperDual object from the expanded TensorHyperDual
        TensorMatHyperDual mat_hyper_dual(r2, d2, h2);

        // Reuse the division logic for TensorMatHyperDual
        return *this / mat_hyper_dual;
    }

    
    /**
     * Overload the / operator for TensorMatHyperDual and torch::Tensor objects.
     *
     * This operator divides a TensorMatHyperDual object by a torch::Tensor. The torch::Tensor
     * is expanded to match the structure of the TensorMatHyperDual, and the dual and hyperdual
     * parts are treated as zero.
     *
     * @param other The torch::Tensor to divide by.
     * @return A new TensorMatHyperDual object representing the element-wise division.
     * @throws std::invalid_argument if the dimensions of the torch::Tensor are incompatible.
     */
    TensorMatHyperDual operator/(const torch::Tensor& other) const {
        // Validate compatibility of dimensions
        if (other.sizes() != this->r.sizes().slice(0, 1).vec()) {
            throw std::invalid_argument("The torch::Tensor must have dimensions compatible with the real part of TensorMatHyperDual.");
        }

        // Expand the torch::Tensor to match TensorMatHyperDual dimensions
        auto r2 = other.unsqueeze(1);                  // Add singleton dimension for batch alignment
        auto d2 = torch::zeros_like(this->d);          // Dual part is zero
        auto h2 = torch::zeros_like(this->h);          // Hyperdual part is zero

        // Reuse the division logic for TensorMatHyperDual
        return *this / TensorMatHyperDual(r2, d2, h2);
    }


    /**
     * Overload the / operator to divide a TensorMatHyperDual object by a scalar (double).
     *
     * This operator performs element-wise division of the real, dual, and hyperdual parts
     * of the TensorMatHyperDual object by the given scalar.
     *
     * @param other The scalar (double) to divide by.
     * @return A new TensorMatHyperDual object with all components divided by the scalar.
     * @throws std::invalid_argument if the scalar is zero.
     */
    TensorMatHyperDual operator/(const double other) const {
        // Handle division by zero
        if (other == 0.0) {
            throw std::invalid_argument("Division by zero is not allowed.");
        }

        // Perform element-wise division for all components
        auto real = this->r / other;
        auto dual = this->d / other;
        auto hyper = this->h / other;

        // Return the scaled TensorMatHyperDual object
        return TensorMatHyperDual(real, dual, hyper);
    }

    /**
     * Index into the TensorMatHyperDual object using a list of TensorIndex objects.
     *
     * This method performs advanced indexing on the real, dual, and hyperdual parts
     * of the TensorMatHyperDual object. If the indexing operation reduces a dimension,
     * it ensures the resulting tensors maintain the expected shape by unsqueezing the
     * missing dimension as needed.
     *
     * @param indices A vector of torch::indexing::TensorIndex objects to specify the indexing.
     * @return A new TensorMatHyperDual object with the indexed components.
     * @throws std::invalid_argument if the indices are invalid.
     */
    TensorMatHyperDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {
        // Index the real part
        auto r = this->r.index(indices);
        if (r.dim() == 2) {  // If a column is missing, unsqueeze dimension 2
            r = r.unsqueeze(2);
        }

        // Index the dual part
        auto d = this->d.index(indices);
        if (d.dim() == 3) {  // If a column is missing, unsqueeze dimension 2
            d = d.unsqueeze(2);
        }

        // Index the hyperdual part
        auto h = this->h.index(indices);
        if (h.dim() == 4) {  // If a column is missing, unsqueeze dimension 2
            h = h.unsqueeze(2);
        }

        // Return a new TensorMatHyperDual object with the indexed components
        return TensorMatHyperDual(r, d, h);
    }


    /**
     * Index into the TensorMatHyperDual object along the first dimension using an integer index.
     *
     * This method extracts the slice corresponding to the specified index from the real, dual,
     * and hyperdual parts of the TensorMatHyperDual object along the batch dimension (first dimension).
     *
     * @param index The integer index to select along the first dimension.
     * @return A new TensorMatHyperDual object representing the selected slice.
     * @throws std::out_of_range if the index is out of bounds.
     */
    TensorMatHyperDual index(int index) const {
        // Validate the index range
        if (index < 0 || index >= this->r.size(0)) {
            throw std::out_of_range("Index out of bounds for the first dimension of TensorMatHyperDual.");
        }

        // Index the real, dual, and hyperdual parts
        auto real = this->r.index({index});
        auto dual = this->d.index({index});
        auto hyper = this->h.index({index});

        // Return the selected slice as a new TensorMatHyperDual object
        return TensorMatHyperDual(real, dual, hyper);
    }

    /**
     * Index into the TensorMatHyperDual object using a boolean mask.
     *
     * This method extracts the elements of the real, dual, and hyperdual parts
     * of the TensorMatHyperDual object based on a boolean mask. The mask must
     * match the shape of the first dimension (batch dimension) of the object.
     *
     * @param mask A boolean torch::Tensor used for indexing.
     * @return A new TensorMatHyperDual object containing the indexed elements.
     * @throws std::invalid_argument if the mask shape is incompatible.
     */
    TensorMatHyperDual index(const torch::Tensor& mask) const {
        // Validate that the mask is a boolean tensor
        if (mask.scalar_type() != torch::kBool) {
            throw std::invalid_argument("The mask must be a boolean tensor.");
        }

        // Validate that the mask matches the size of the first dimension
        if (mask.sizes() != this->r.sizes().slice(0, 1).vec()) {
            throw std::invalid_argument("The mask must match the size of the first dimension of TensorMatHyperDual.");
        }

        // Perform indexing on the real, dual, and hyperdual parts
        auto real = r.index({mask});
        auto dual = d.index({mask});
        auto hyper = h.index({mask});

        // Return a new TensorMatHyperDual object with the indexed components
        return TensorMatHyperDual(real, dual, hyper);
    }

    /**
     * Index into the TensorMatHyperDual object using a vector of TensorIndex objects.
     *
     * This method performs advanced indexing on the real, dual, and hyperdual parts
     * of the TensorMatHyperDual object using a vector of torch::indexing::TensorIndex.
     * The indexing operation allows for slicing, masking, or selecting specific elements.
     *
     * @param indices A vector of torch::indexing::TensorIndex specifying the indexing.
     * @return A new TensorMatHyperDual object containing the indexed components.
     * @throws std::invalid_argument if the indexing fails or is incompatible.
     */
    TensorMatHyperDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {
        try {
            // Perform indexing on the real, dual, and hyperdual parts
            auto real = r.index(indices);
            auto dual = d.index(indices);
            auto hyper = h.index(indices);

            // Return a new TensorMatHyperDual object with the indexed components
            return TensorMatHyperDual(real, dual, hyper);
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("Indexing failed: ") + e.what());
        }
    }

    /**
     * Enable or disable gradient computation for the TensorMatHyperDual object.
     *
     * This method sets the `requires_grad` attribute for the real, dual, and hyperdual
     * tensors of the TensorMatHyperDual object. It allows toggling the gradient tracking
     * behavior for all components.
     *
     * @param req_grad A boolean value indicating whether gradients should be tracked.
     */
    void requires_grad_(bool req_grad) {
        r.requires_grad_(req_grad);
        d.requires_grad_(req_grad);
        h.requires_grad_(req_grad);
    }

    /**
     * Compute gradients for the TensorMatHyperDual object.
     *
     * This method calls the `backward()` function on the real, dual, and hyperdual tensors.
     * It ensures that gradient computation is only triggered for tensors with `requires_grad` enabled.
     *
     * @throws std::runtime_error if any tensor has more than one element and no gradient is specified.
     */
    void backward(const torch::optional<torch::Tensor>& grad_r = {}, 
                const torch::optional<torch::Tensor>& grad_d = {}, 
                const torch::optional<torch::Tensor>& grad_h = {}) {
        // Check if gradients are required
        if (r.requires_grad()) {
            if (grad_r) {
                r.backward(grad_r.value());
            } else if (r.numel() == 1) {
                r.backward();
            } else {
                throw std::runtime_error("Gradient for 'r' must be specified when it has more than one element.");
            }
        }

        if (d.requires_grad()) {
            if (grad_d) {
                d.backward(grad_d.value());
            } else if (d.numel() == 1) {
                d.backward();
            } else {
                throw std::runtime_error("Gradient for 'd' must be specified when it has more than one element.");
            }
        }

        if (h.requires_grad()) {
            if (grad_h) {
                h.backward(grad_h.value());
            } else if (h.numel() == 1) {
                h.backward();
            } else {
                throw std::runtime_error("Gradient for 'h' must be specified when it has more than one element.");
            }
        }
    }




    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     */
    static TensorHyperDual einsum(const std::string& arg, 
                                  const TensorMatHyperDual& first, 
                                  const TensorHyperDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");

        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);


        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1 = torch::einsum(darg1, {first.d,  second.r});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        //Now for the hyper dual part
        auto d1d2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1d2 = torch::einsum(d1d2arg, {first.d,  second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h,  second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r,  second.h});
        auto h = d1d2.unsqueeze(-1) + h1r2 + r1h2;



        return TensorHyperDual(std::move(r), std::move(d1 + d2), std::move(h));
    }

    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     * @param arg The einsum string specifying the operation.
     * @param first The first TensorDual object.
     */
    static TensorMatHyperDual einsum(const std::string& arg, 
                                     const TensorMatHyperDual& first, 
                                     const torch::Tensor& second) {

        auto r = torch::einsum(arg, {first.r, second});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        //The result here should be 

        auto d1r1arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d = torch::einsum(d1r1arg, {first.d,  second});

        auto h1r1arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h = torch::einsum(h1r1arg, {first.h,  second});

        return TensorMatHyperDual(std::move(r), std::move(d), std::move(h));
    }

    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     * @param arg The einsum string specifying the operation.
     * @param first The first TensorDual object.
     * @param second The second TensorDual object.
     * @return A new TensorDual object representing the result of the einsum operation.
     */
    static TensorMatHyperDual einsum(const std::string& arg, 
                                     const TensorHyperDual& first, 
                                     const TensorMatHyperDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        

        auto d1r2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1r2 = torch::einsum(d1r2arg, {first.d,  second.r});
        auto r1d2arg = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto r1d2 = torch::einsum(r1d2arg, {first.r, second.d});

        auto d1d2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1d2 = torch::einsum(d1d2arg, {first.d,  second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h,  second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r,  second.h});
        auto h = d1d2.unsqueeze(-1) + h1r2 + r1h2;

        return TensorMatHyperDual(std::move(r), std::move(d1r2 + r1d2), std::move(h));
    }
    
    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     * @param arg The einsum string specifying the operation.
     * @param first The first TensorDual object.
     * @param second The second TensorDual object.
     * @return A new TensorDual object representing the result of the einsum operation.
     */
    static TensorMatHyperDual einsum(const std::string& arg, 
                                     const torch::Tensor& first, 
                                     const TensorMatHyperDual& second) {

        auto r = torch::einsum(arg, {first, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);


        auto r1d1arg = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto r1d1 = torch::einsum(r1d1arg, {first,  second.d});

        auto r1h1arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";

        auto r1h1 = torch::einsum(r1h1arg, {first,  second.h});

        return TensorMatHyperDual(std::move(r), std::move(r1d1), std::move(r1h1));
    }


    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     * @param arg The einsum string specifying the operation.
     * @param first The first TensorDual object.
     * @param second The second TensorDual object.
     * @return A new TensorDual object representing the result of the einsum operation.
     * @throws std::invalid_argument if the dimensions of the two objects are incompatible.
     */
    static TensorMatHyperDual einsum(const std::string& arg, 
                                     const TensorMatHyperDual& first, 
                                     const TensorMatHyperDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);

        auto r1d2arg = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto r1d2 = torch::einsum(r1d2arg, {first.r,  second.d});
        auto d1r2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1r2 = torch::einsum(d1r2arg, {first.d, second.r});

        auto d1d2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1d2 = torch::einsum(d1d2arg, {first.d,  second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h,  second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r,  second.h});

        return TensorMatHyperDual(std::move(r), 
                                  std::move(r1d2 + d1r2), 
                                  std::move(2*d1d2.unsqueeze(-1) + h1r2 + r1h2));
    }

    /**
     * In-place assignment to the TensorMatHyperDual object using a mask and a TensorHyperDual value.
     *
     * This method updates the elements of the real, dual, and hyperdual parts of the
     * TensorMatHyperDual object in-place, using a boolean mask and values from a
     * TensorHyperDual object. The mask determines which elements to update.
     *
     * @param mask A boolean torch::Tensor specifying the elements to update.
     * @param value A TensorHyperDual object whose values will be assigned to the masked elements.
     * @throws std::invalid_argument if the mask or value dimensions are incompatible.
     */
    void index_put_(const torch::Tensor& mask, const TensorHyperDual& value) {
        // Validate that the mask is a boolean tensor
        if (mask.scalar_type() != torch::kBool) {
            throw std::invalid_argument("The mask must be a boolean tensor.");
        }

        // Validate that the dimensions of the value match the broadcasted dimensions of the masked elements
        if (value.r.sizes() != this->r.sizes().slice(1).vec() ||
            value.d.sizes() != this->d.sizes().slice(1).vec() ||
            value.h.sizes() != this->h.sizes().slice(1).vec()) {
            throw std::invalid_argument("The value dimensions must match the broadcasted dimensions of the masked elements.");
        }

        // Perform in-place updates on the real, dual, and hyperdual parts
        this->r.index_put_({mask}, value.r.squeeze());
        this->d.index_put_({mask}, value.d.squeeze());
        this->h.index_put_({mask}, value.h.squeeze());
    }

    /**
     * In-place assignment to the TensorMatHyperDual object using a TensorIndex mask and a TensorHyperDual value.
     *
     * This method updates the elements of the real, dual, and hyperdual parts of the
     * TensorMatHyperDual object in-place, using a TensorIndex mask and values from a
     * TensorHyperDual object.
     *
     * @param mask A TensorIndex specifying the elements to update.
     * @param value A TensorHyperDual object whose values will be assigned to the masked elements.
     * @throws std::invalid_argument if the mask or value dimensions are incompatible.
     */
    void index_put_(const torch::indexing::TensorIndex& mask, const TensorHyperDual& value) {
        // Ensure that the value dimensions are compatible
        if (value.r.dim() != this->r.dim() - 1 || 
            value.d.dim() != this->d.dim() - 1 || 
            value.h.dim() != this->h.dim() - 1) {
            throw std::invalid_argument("The value dimensions must be compatible with the TensorMatHyperDual structure.");
        }

        // Perform in-place updates on the real, dual, and hyperdual parts
        this->r.index_put_({mask}, value.r.squeeze());
        this->d.index_put_({mask}, value.d.squeeze());
        this->h.index_put_({mask}, value.h.squeeze());
    }

    /**
     * In-place assignment to the TensorMatHyperDual object using a vector of TensorIndex and a TensorMatHyperDual value.
     *
     * This method updates the elements of the real, dual, and hyperdual parts of the
     * TensorMatHyperDual object in-place using a vector of TensorIndex and values from
     * another TensorMatHyperDual object.
     *
     * @param mask A vector of TensorIndex specifying the elements to update.
     * @param value A TensorMatHyperDual object whose values will be assigned to the masked elements.
     * @throws std::invalid_argument if the mask or value dimensions are incompatible.
     */
    void index_put_(const std::vector<torch::indexing::TensorIndex>& mask, const TensorMatHyperDual& value) {
        // Validate that the dimensions of the value match the expected shape for the masked elements
        auto r_masked_shape = this->r.index(mask).sizes();
        if (value.r.sizes() != r_masked_shape || 
            value.d.sizes() != this->d.index(mask).sizes() || 
            value.h.sizes() != this->h.index(mask).sizes()) {
            throw std::invalid_argument("The value dimensions must match the shape of the selected elements in TensorMatHyperDual.");
        }

        // Perform in-place updates on the real, dual, and hyperdual parts
        this->r.index_put_(mask, value.r);
        this->d.index_put_(mask, value.d);
        this->h.index_put_(mask, value.h);
    }

    /**
     * In-place assignment to the TensorMatHyperDual object using a vector of TensorIndex and a scalar value.
     *
     * This method updates the elements of the real part of the TensorMatHyperDual object
     * with a scalar value while setting the corresponding dual and hyperdual parts to zero.
     *
     * @param mask A vector of TensorIndex specifying the elements to update.
     * @param value A scalar value (double) to assign to the real part.
     */
    void index_put_(const std::vector<torch::indexing::TensorIndex>& mask, const double& value) {
        // Update the real part with the scalar value
        this->r.index_put_({mask}, value);

        // Set the corresponding dual and hyperdual parts to zero
        this->d.index_put_({mask}, 0.0);
        this->h.index_put_({mask}, 0.0);
    }

    
    /**
     * In-place assignment to the TensorMatHyperDual object using a vector of TensorIndex and a torch::Tensor value.
     *
     * This method updates the elements of the real part of the TensorMatHyperDual object
     * using a given torch::Tensor value, while setting the corresponding dual and hyperdual
     * parts to zero.
     *
     * @param mask A vector of TensorIndex specifying the elements to update.
     * @param value A torch::Tensor specifying the values to assign to the real part.
     * @throws std::invalid_argument if the dimensions of the value tensor do not match the selected elements.
     */
    void index_put_(const std::vector<torch::indexing::TensorIndex>& mask, const torch::Tensor& value) {
        // Validate that the value tensor's dimensions match the shape of the selected elements
        auto r_masked_shape = this->r.index(mask).sizes();
        if (value.sizes() != r_masked_shape) {
            throw std::invalid_argument("The value tensor's dimensions must match the shape of the selected elements in the real part.");
        }

        // Update the real part with the given tensor value
        this->r.index_put_({mask}, value);

        // Set the corresponding dual and hyperdual parts to zero
        this->d.index_put_({mask}, 0.0);
        this->h.index_put_({mask}, 0.0);
    }

    /**
     * Convert a TensorHyperDual to a TensorMatHyperDual by adding a singleton dimension.
     *
     * This method takes a TensorHyperDual object and creates a TensorMatHyperDual object
     * by unsqueezing a specified dimension in the real, dual, and hyperdual parts.
     *
     * @param x The TensorHyperDual object to be converted.
     * @param dim The dimension to unsqueeze (add a singleton dimension).
     * @return A TensorMatHyperDual object with the specified dimension unsqueezed.
     * @throws std::invalid_argument if the dimension is out of range.
     */
    static TensorMatHyperDual unsqueeze(const TensorHyperDual& x, int dim) {
        // Validate that the dimension is within the valid range for unsqueezing
        if (dim < 0 || dim > x.r.dim()) {
            throw std::invalid_argument("Dimension out of range for unsqueeze operation.");
        }

        // Unsqueeze the real, dual, and hyperdual parts
        auto r = x.r.unsqueeze(dim);
        auto d = x.d.unsqueeze(dim);
        auto h = x.h.unsqueeze(dim);

        // Create and return a TensorMatHyperDual object
        return TensorMatHyperDual(r, d, h);
    }


};

TensorMatHyperDual TensorHyperDual::eye() {
        auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});
        auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(2), this->d.size(3)}, this->d.options());
        auto h = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(2), this->d.size(3), this->h.size(4)}, this->h.options());
        return TensorMatHyperDual(r, d, h);
}


TensorMatDual TensorDual::unsqueeze(int dim)
{
        auto r = this->r.unsqueeze(dim);
        auto d = this->d.unsqueeze(dim);
        return TensorMatDual(std::move(r), std::move(d));
}

TensorMatDual TensorDual::eye() {
        auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});
        auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(1), this->d.size(2)}, this->d.options());
        return TensorMatDual(r, d);
}


// Non-member overload for torch::Tensor * TensorDual
TensorDual operator*(const torch::Tensor& tensor, const TensorDual& td) {
    auto real = tensor * td.r;
    auto dual = tensor.unsqueeze(-1) * td.d;
    return TensorDual(std::move(real), std::move(dual));
}


// Non-member overload for torch::Tensor * TensorDual
TensorHyperDual operator*(const torch::Tensor& tensor, const TensorHyperDual& td) {
    auto real = tensor * td.r;
    auto dual = tensor.unsqueeze(-1) * td.d;
    auto hyper = tensor.unsqueeze(-1).unsqueeze(-1) * td.h;
    return TensorHyperDual(std::move(real), std::move(dual), std::move(hyper));
}


// Non-member overload for torch::Tensor / TensorDual
TensorDual operator/(const torch::Tensor& tensor, const TensorDual& td) {
    auto r = tensor / td.r;
    auto d = -(tensor / td.r.square()).unsqueeze(-1) * td.d;
    return TensorDual(r, d);
}



// Non-member overload for torch::Tensor / TensorDual
TensorHyperDual operator/(const torch::Tensor& tensor, const TensorHyperDual& td) {
    auto r1 = tensor;
    auto r2 = td.r;
    auto d2 = td.d;
    auto h2 = td.h;
    auto r = r1 / r2;
    auto d = -(r1 / r2.square()).unsqueeze(-1) * d2;
    auto h = torch::einsum("mi, mi, mij, mik->mijk",{r1, r2.pow(-3), d2, d2}) - 
             torch::einsum("mi, mi, mijk->mijk",{r1, r2.pow(-2), h2});
    return TensorHyperDual(r, d, h);
}

// Non-member overload for torch::Tensor + TensorDual
TensorDual operator+(const torch::Tensor& tensor, const TensorDual& td) {
    return TensorDual(std::move(tensor + td.r), std::move(td.d.clone()));
}



// Non-member overload for torch::Tensor + TensorDual
TensorHyperDual operator+(const torch::Tensor& tensor, const TensorHyperDual& td) {
    return TensorHyperDual(std::move(tensor + td.r), std::move(td.d.clone()), std::move(td.h.clone()));
}

// Non-member template function for Scalar + TensorDual
TensorDual operator+(const double& scalar, const TensorDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorDual((scalar + td.r).clone(), td.d.clone());
}



// Non-member template function for Scalar + TensorDual
TensorHyperDual operator+(const double& scalar, const TensorHyperDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorHyperDual((scalar + td.r).clone(), td.d.clone(), td.h.clone());
}

// Non-member overload for torch::Tensor - TensorDual
TensorDual operator-(const torch::Tensor& tensor, const TensorDual& td) {
    return TensorDual(std::move(tensor - td.r), std::move(-td.d.clone()));
}



// Non-member overload for torch::Tensor - TensorDual
TensorHyperDual operator-(const torch::Tensor& tensor, const TensorHyperDual& td) {
    return TensorHyperDual(std::move(tensor - td.r), std::move(-td.d.clone()), std::move(-td.h.clone()));
}



TensorDual operator*(const TensorDual& td, const TensorMatDual& other)  {

    torch::Tensor r, d;
    if ( td.r.size(1) ==  other.r.size(1) ) {
        //Left multiply
        r = torch::einsum("mi, mij->mj", {td.r, other.r});
        d = torch::einsum("mi, mijn->mjn", {td.r, other.d}) +
            torch::einsum("min, mij->mjn", {td.d, other.r});
    }
    else 
    {
        //Right multiply
        r = torch::einsum("mi, mji->mj", {td.r, other.r});
        d = torch::einsum("mi, mjin->mjn", {td.r, other.d})+
            torch::einsum("min, mji->mjn", {td.d, other.r});
    }
    return TensorDual(r, d);
}

TensorDual operator*(const TensorMatDual& tmd, const TensorDual& other)  {
    torch::Tensor r,d;
    if ( tmd.r.size(2) ==  other.r.size(1) ) 
    {
      r = torch::einsum("mij, mj->mi",{tmd.r, other.r});
      d = torch::einsum("mijn, mj->min", {tmd.d, other.r}) +
          torch::einsum("mij, mjn->min", {tmd.r, other.d});
    }
    else
    {
      r = torch::einsum("mij, mi->mj",{tmd.r, other.r});
      d = torch::einsum("mijn, mi->mjn", {tmd.d, other.r}) +
          torch::einsum("mij, min->mjn", {tmd.r, other.d});

    }
    return TensorDual(r, d);
}

TensorMatDual operator*(const TensorMatDual& lhs, const TensorMatDual& rhs) {
    auto r = lhs.r * rhs.r;
    auto d = torch::einsum("mij, mijn->mijn", {lhs.r, rhs.d}) + torch::einsum("mijn, mij->mijn", {lhs.d, rhs.r});
    return TensorMatDual(std::move(r), std::move(d));
}



// Non-member function to handle scalar - TensorDual
TensorDual operator-(const int& scalar, const TensorDual& td) {
    auto scalar_tensor = torch::tensor({scalar}, td.r.options()); // Create a tensor filled with 'scalar'
    return TensorDual(scalar_tensor - td.r, -td.d);
}



// Non-member template function for Scalar / TensorDual
TensorDual operator/(double& scalar, const TensorDual& td) {
    auto r = scalar/td.r;
    auto d = -(scalar/(td.r.square())).unsqueeze(-1) * td.d;
    return TensorDual(r, d);
}



//overload the * operator for a double and a TensorDual
TensorDual operator*(const double& scalar, const TensorDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorDual(td.r * scalar, td.d * scalar);
}

//overload the * operator for a double and a TensorDual
TensorMatDual operator*(const double& scalar, const TensorMatDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorMatDual(td.r * scalar, td.d * scalar);
}


//pow for TensorDual to a TensorDual
TensorDual pow(const TensorDual& base, const TensorDual& exponent) {
    //If x = a+b \epsilon and y = c+d \epsilon, then
    //x^y = a^c+(a^(c-1)*b*c+a^c*d*log(a))\epsilon
        auto a = base.r;
        auto b = base.d;
        auto c = exponent.r;
        auto d = exponent.d;
        auto real = torch::pow(a, c);
        auto dual = torch::einsum("mi, mij->mij", {real * torch::log(a), d}) + 
                    torch::einsum("mi, mij->mij", {real * (c / a), b});
        return TensorDual(real, dual);
}

//overload the pow method for a TensorDual and a scalar
TensorDual pow(const TensorDual&base, const double& exponent) {
    auto real = torch::pow(base.r, exponent);
    auto dual = torch::einsum("mi, mij->mij", {exponent * torch::pow(base.r, exponent - 1), base.d});
    //std::cerr << "dual sizes in ^ with scalar: " << dual.sizes() << std::endl;
    return TensorDual(real, dual);
}

TensorDual max(const TensorDual& lhs, const TensorDual& rhs) {
    auto r = torch::max(lhs.r, rhs.r);
    auto d = torch::zeros_like(lhs.d);
    auto maskrgt = lhs.r < rhs.r;
    d.index_put_({maskrgt}, rhs.d.index({maskrgt}));
    d.index_put_({~maskrgt}, lhs.d.index({~maskrgt}));
    return TensorDual(r, d);
}

/*TensorDual max(const torch::Tensor& lhs, const TensorDual& rhs) {
    auto lhsd = TensorDual::toDual(lhs, rhs);
    return max(lhsd, rhs);
}*/

TensorDual max(const TensorDual& lhs, const torch::Tensor& rhs) {
    auto mask = lhs.r > rhs;
    auto resr = torch::zeros_like(lhs.r);
    auto resd = torch::zeros_like(lhs.d);    
    resr.index_put_({mask}, lhs.r.index({mask}));
    rhs.dim() == lhs.r.dim() ? resr.index_put_({~mask}, rhs.index({~mask})) : 
                               resr.index_put_({~mask}, rhs);
    resd.index_put_({mask}, lhs.d.index({mask}));
    return TensorDual(resr, resd);
}




TensorDual min(const TensorDual& lhs, const TensorDual& rhs) {
    //Check to see if we are taking the minimum of an empty tensor
    auto r = torch::min(lhs.r, rhs.r);
    auto maskl = lhs.r < rhs.r;
    auto d = torch::zeros_like(lhs.d);
    d.index_put_({maskl}, lhs.d.index({maskl}));
    d.index_put_({~maskl}, rhs.d.index({~maskl}));
    return TensorDual(r, d);
}



TensorDual min(const TensorDual& lhs, const torch::Tensor& rhs) {
    auto mask = lhs.r < rhs;
    auto resr = torch::zeros_like(lhs.r);
    auto resd = torch::zeros_like(lhs.d);
    resr.index_put_({mask}, lhs.r.index({mask}));
    resr.index_put_({~mask}, rhs.index({~mask}));
    resd.index_put_({mask}, lhs.d.index({mask}));
    return TensorDual(resr, resd);
}


static TensorDual sign(TensorDual& td) {
    auto r = torch::sign(td.r);
    auto maskz = td.r == 0;
    auto d = torch::zeros_like(td.d);
    if ( maskz.any().item<bool>())
    {
      d.index_put_({maskz}, r.index({maskz}));//The dual part is the same as the sign only if x==0
    }
    return TensorDual(r, d);
}






// ... [Other parts of the TensorDual class]

TensorMatDual ger(const TensorDual& x, const TensorDual& y) {

        // Replicate einsum 'mj, mi->mij'
        auto r = torch::einsum("mj, mi->mij", {x.r, y.r});

        // Replicate einsum 'mj, mik->mijk' and 'mjk, mi->mijk'
        //        d1 = torch.einsum('mj, mik->mijk', x.r, y.d)
        auto d1 = torch::einsum("mj, mik->mijk", {x.r, y.d});
        //d2  = torch.einsum('mjk, mi->mijk', x.d, y.r)
        auto d2 = torch::einsum("mjk, mi->mijk", {x.d, y.r});

        // Create a TensorMatDual from the results
        return TensorMatDual(r, d1 + d2);
}


TensorDual defaultTensorDual = TensorDual::createZero(std::move(torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kFloat64))), 1);

}



#endif