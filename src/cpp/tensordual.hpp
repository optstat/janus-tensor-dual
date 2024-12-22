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
            throw std::runtime_error("The real tensor `r` must be 2D.");
        }
        if (d.dim() != 3) {
            throw std::runtime_error("The dual tensor `d` must be 3D.");
        }
        if (r.device() != d.device()) {
            throw std::runtime_error("Real and dual tensors must reside on the same device.");
        }
        this->r = r;
        this->d = d;
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
        // Validate tensor dimensions
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Tensors in TensorDual must have the same shape for division.");
        }

        // Ensure the denominator is safe for division
        auto safe_r = torch::sign(other.r) * other.r.abs().clamp_min(1e-12);
    
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
        auto safe_other = torch::sign(other) * other.abs().clamp_min(1e-12);


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
        auto safe_scalar = torch::sign(tscalar) * tscalar.abs().clamp_min(1e-12);

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
        auto r_safe = torch::sign(r) * r.abs().clamp_min(1e-12);
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
    : r(torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      d(torch::zeros({1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
      h(torch::zeros({1, 1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
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
        if (r.dim() != 2) throw std::invalid_argument("Real part must be a 2D tensor");
        if (d.dim() != 3) throw std::invalid_argument("Dual part must be a 3D tensor");
        if (h.dim() != 4) throw std::invalid_argument("Hyperdual part must be a 4D tensor");
        if (r.device() != d.device() || d.device() != h.device()) {
            throw std::invalid_argument("All tensors must reside on the same device");
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
        h(torch::zeros({x.d.size(0), x.d.size(1), x.d.size(2), x.d.size(2)}, x.d.options())),
        dtype_(torch::typeMetaToScalarType(x.r.dtype())),
        device_(x.r.device()) {
            validateTensors(r, d, h);
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
     * @return A new TensorHyperDual object with summed components.
     * 
     * @note Assumes that `r`, `d`, and `h` are tensors with compatible dimensions.
     */
    TensorHyperDual sum() const {
        auto r = this->r.sum(1, true);
        auto d = this->d.sum(1, true);
        auto h = this->h.sum(1, true);
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
      auto rc = r;
      auto dc = d;
      auto hc = h;
      //If there are negative numbers convert to complex
      if ((r < 0).any().item<bool>() && (r.is_complex() == false)) {
            rc = torch::complex(r, torch::zeros_like(r).to(r.device()));
            dc = torch::complex(d, torch::zeros_like(d).to(d.device()));
            hc = torch::complex(h, torch::zeros_like(h).to(h.device()));

      }

      auto rn = torch::sqrt(rc);

      // Compute the dual part
      auto dn = 0.5 * torch::einsum("mi, mij->mij", {rc.pow(-0.5), dc});

      // Compute the hyperdual part
      auto hn = -0.25 * torch::einsum("mi, mij, mik->mijk", {rc.pow(-1.5), dc, dc}) +
                0.5 * torch::einsum("mi, mijk->mijk", {rc.pow(-0.5), hc});

      // Return the resulting TensorHyperDual object
      return TensorHyperDual(rn, dn, hn);
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
    //If the number is really small add a tiny number to avoid division by zero
    dn = torch::where(dn.abs() < 1e-14, torch::ones_like(dn) * 1e-14, dn);

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


        // Return the mask for batch selection
        return mask;
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


        return mask;
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


        return mask;

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

        return mask;
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

        return mask;
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
        return mask;
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
        return mask;
    }

    /**
     * Overload the less-than operator for TensorHyperDual < TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator<(const TensorHyperDual& other) const {
        auto mask = r < other.r;
        return mask;
    }

    /**
     * Overload the less-than operator for TensorHyperDual < torch::Tensor.
     * This also implicitly supports TensorHyperDual < Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator<(const torch::Tensor& other) const {
        auto mask = r < other;
        return mask;
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
        return mask;
    }



    /**
     * Overload the greater-than-or-equal-to operator for TensorHyperDual >= TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator>=(const TensorHyperDual& other) const {
        auto mask = r >= other.r;
        return mask;
    }

    /**
     * Overload the greater-than-or-equal-to operator for TensorHyperDual >= torch::Tensor.
     * This also implicitly supports TensorHyperDual >= Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator>=(const torch::Tensor& other) const {
        auto mask = r >= other;
        return mask;
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
        return mask;
    }



    /**
     * Overload the equality operator for TensorHyperDual == torch::Tensor.
     * This also implicitly supports TensorHyperDual == Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator==(const torch::Tensor& other) const {
        auto mask = r.eq(other); // Using the .eq() function for equality comparison
        return mask;
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
        return mask;
    }

    /**
     * Overload the inequality operator for TensorHyperDual != TensorHyperDual.
     * @param other The TensorHyperDual object to compare with.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator!=(const TensorHyperDual& other) const {
        auto mask = r.ne(other.r); // Using the .ne() function for inequality comparison
        return mask;
    }

    /**
     * Overload the inequality operator for TensorHyperDual != torch::Tensor.
     * This also implicitly supports TensorHyperDual != Scalar due to PyTorch's implicit scalar handling.
     * @param other A torch::Tensor to compare with the real part (r) of TensorHyperDual.
     * @return A torch::Tensor boolean mask indicating where the real part of this TensorHyperDual
     */
    torch::Tensor operator!=(const torch::Tensor& other) const {
        auto mask = r.ne(other); // Using the .ne() function for inequality comparison
        return mask;
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
        return mask;
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
        torch::Tensor rn, dn, hn;
        if (this->r.is_complex()) {
            rn = torch::imag(this->r);
        }
        else 
        {
            rn = torch::zeros_like(this->r);
        }
        if (this->d.is_complex()) {
            dn = torch::imag(this->d);
        }
        else 
        {
            dn = torch::zeros_like(this->d);
        }
        if (this->h.is_complex()) {
            hn = torch::imag(this->h);
        }
        else 
        {
            hn = torch::zeros_like(this->h);
        }
        return TensorHyperDual(rn, dn, hn);
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
        torch::Tensor sign_r;
        if (this->r.is_complex()) {
            sign_r = torch::sgn(this->r);
        }
        else {
            sign_r = torch::sign(this->r);
        }

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
        auto h = h1r2;

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
        auto rhsarg = arg.substr(posa+2);

        //Find the positions of the "," in the einsum string
        std::vector<std::string> lhsargs{};
        size_t pos = arg.find(',');
        size_t posl = 0;
        while(pos != std::string::npos) 
        {
           lhsargs.push_back(arg.substr(posl, pos-posl));
           //Find the next comma
           posl = pos+1;
           pos = arg.find(',', pos+1);
        }
        //The last argument is before the ->
        lhsargs.push_back(arg.substr(posl, posa-posl));
        rhsarg = arg.substr(posa+2);

        auto d = torch::zeros_like(tensors[0].d);
        for ( int i=0; i < tensors.size(); i++)
        {
            std::vector<torch::Tensor> dpart = {torch::zeros_like(tensors[0].d)};
            auto dl = tensors[i].d;
            auto darg = lhsargs[i] + "z";
            for ( int j=0; j < tensors.size(); j++)
            {
                if ( i==j) continue;
                darg +=  ","+lhsargs[j];
                dpart.push_back(tensors[j].r);
            }
            dl = dl+torch::einsum(darg, dpart);

            d = d + dl;
        }

        torch::Tensor h = torch::zeros_like(tensors[0].h);
        for ( int i=0; i < tensors.size(); i++)
        {
            //For each h there is one h and the rest are all are real parts
            //After that we have pairs of dual parts while the rest are real parts
            std::vector<torch::Tensor> hpart = {tensors[i].h};
            std::string harg = lhsargs[i] + "zw";
            for ( int j=0; j < tensors.size(); j++)
            {
                if ( i==j) continue;
                harg +=  ","+lhsargs[j];
                hpart.push_back(tensors[j].r);
            }
            harg = harg + "->" + rhsarg + "zw";
            h = h + torch::einsum(harg, hpart);
        }
        //We now need to evaluate the pairs of dual parts
        for ( int i=0; i < tensors.size(); i++)
        {
            //First dual part
            auto d1 = tensors[i].d;
            std::string dharg = lhsargs[i] + "z";
            std::vector<torch::Tensor> dhpart = {tensors[i].d};
            for ( int j=0; j < tensors.size(); j++)
            {
                if ( i==j) continue;
                //second dual part
                dharg +=  ","+lhsargs[j]+"w";
                dhpart.push_back(tensors[j].d);
                //Finally the real parts
                for ( int k=0; k < tensors.size(); k++)
                {
                    if ( i==k ||  j==k) continue;
                    dharg +=  ","+lhsargs[k];
                    dhpart.push_back(tensors[k].r);
                }
            }
            dharg = dharg + "->" + rhsarg + "zw";
            h = h+torch::einsum(dharg, dhpart);
        }
        
        return TensorHyperDual(r, d, h);
    }



};



/**
 * @brief The TensorMatDual class tracks sensitivities (first-order derivatives) of a matrix.
 * 
 * This class extends the concept of TensorDual to matrices, keeping track of first-order
 * derivatives with respect to initial conditions. It is designed for use cases where
 * matrix-valued functions are involved, and their sensitivities need to be computed and stored.
 */
class TensorMatDual {
public:
    // Members
    torch::Tensor r;                    ///< Real part (matrix)
    torch::Tensor d;                    ///< Dual part (first-order derivatives)
    torch::Dtype dtype_ = torch::kFloat64; ///< Data type of the tensors
    torch::Device device_ = torch::kCPU;   ///< Device where the tensors reside (CPU or GPU)

    // Constructor
    /**
     * @brief Constructs a TensorMatDual object.
     * 
     * @param r A 3D tensor representing the real part (matrix).
     * @param d A 4D tensor representing the dual part (first-order derivatives).
     * @throws std::invalid_argument If the dimensions of `r` or `d` are incorrect.
     */
    TensorMatDual(torch::Tensor r, torch::Tensor d) {
        if (r.dim() != 3) {
            throw std::invalid_argument("In TensorMatDual, the real part must be a 3D tensor.");
        }
        if (d.dim() != 4) {
            throw std::invalid_argument("Dual part of TensorMatDual must be a 4D tensor.");
        }
        if (r.device() != d.device()) {
            throw std::invalid_argument("Real and dual tensors must reside on the same device.");
        }

        this->r = std::move(r);
        this->d = std::move(d);
        this->dtype_ = torch::typeMetaToScalarType(this->r.dtype());
        this->device_ = this->r.device();
    }

    // Default Constructor
    /**
     * @brief Default constructor initializes the TensorMatDual object with empty tensors.
     * 
     * This constructor creates a `TensorMatDual` object with uninitialized tensors for the
     * real (`r`) and dual (`d`) parts. The tensors have zero dimensions and are initialized
     * on the default device (`kCPU`) with the default data type (`kFloat64`).
     */
    TensorMatDual()
        : r(torch::empty({0, 0, 0}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
        d(torch::empty({0, 0, 0, 0}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))) {}


    // Copy Constructor
    /**
     * @brief Constructs a new TensorMatDual object as a deep copy of another TensorMatDual object.
     * 
     * This constructor performs a deep copy of the real (`r`) and dual (`d`) tensors from
     * the given `other` TensorMatDual object. The tensors are cloned, ensuring that changes
     * to the new object do not affect the original.
     * 
     * @param other The TensorMatDual object to copy from.
     */
    TensorMatDual(const TensorMatDual& other) {
        // Clone the real and dual tensors to ensure deep copy
        this->r = other.r.clone();
        this->d = other.d.clone();

        // Copy the device and data type settings
        this->device_ = other.device_;
        this->dtype_ = other.dtype_;
    }

    /**
     * @brief Moves the TensorMatDual object to a specified device in place.
     * 
     * This method modifies the current TensorMatDual object by moving its tensors
     * to the specified device.
     * 
     * @param device The target torch::Device to move the tensors to.
     * @return A reference to the modified TensorMatDual object.
     */
    TensorMatDual& to_(torch::Device device) {
        // Move tensors to the specified device
        this->r = this->r.to(device);
        this->d = this->d.to(device);
        this->device_ = device;
        return *this;
    }

    /**
     * @brief Retrieves the device on which the tensor is stored.
     *
     * This method provides access to the `torch::Device` associated with the tensor. 
     * It allows users to query the device (e.g., CPU, CUDA) that the tensor 
     * resides on, which can be useful for ensuring operations occur on the 
     * correct hardware.
     *
     * @return torch::Device The device associated with this tensor.
     */
    torch::Device device() const {
        return this->device_;
    }


    /**
     * @brief Constructs a TensorMatDual object by unsqueezing the provided TensorDual along a specified dimension.
     *
     * @param x The input TensorDual containing the real and dual parts.
     * @param dim The dimension along which to unsqueeze (default is 2).
     * @throws std::invalid_argument If the specified dimension is invalid for the input tensors.
     */
    TensorMatDual(const TensorDual& x, int dim = 2)
        : TensorMatDual(
            x.r.dim() >= dim ? x.r.unsqueeze(dim) : throw std::invalid_argument("Invalid dimension for unsqueeze"),
            x.d.dim() >= dim ? x.d.unsqueeze(dim) : throw std::invalid_argument("Invalid dimension for unsqueeze")
        ) {}

    /**
     * @brief Converts the real and dual parts of the TensorMatDual to complex tensors.
     *
     * If the real (`r`) or dual (`d`) part is not already in a complex format, 
     * they will be converted by pairing them with zero tensors as the imaginary component.
     *
     * @return TensorMatDual A new TensorMatDual object with complex real and dual parts.
     */
    TensorMatDual complex() const {
        // Convert real part to complex if not already complex
        torch::Tensor rc = this->r.is_complex() 
            ? this->r 
            : torch::complex(this->r, torch::zeros_like(this->r)).to(this->device_);
        
        // Convert dual part to complex if not already complex
        torch::Tensor dc = this->d.is_complex() 
            ? this->d 
            : torch::complex(this->d, torch::zeros_like(this->d)).to(this->device_);
        
        // Return a new TensorMatDual object
        return TensorMatDual(rc, dc);
    }

    /**
     * @brief Overloads the output stream operator for the TensorMatDual class.
     *
     * This method enables printing the TensorMatDual object using standard
     * output streams (e.g., `std::cout` or `std::ostream`). It outputs:
     * - The real part (r) of the TensorMatDual object.
     * - The dual part (d) of the TensorMatDual object.
     * - The shapes of both tensors for additional context.
     *
     * Example Usage:
     * @code
     * TensorMatDual obj = ...;
     * std::cout << obj << std::endl;
     * @endcode
     *
     * @param os The output stream to which the TensorMatDual object is written.
     * @param obj The TensorMatDual object to be output.
     * @return std::ostream& The modified output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const TensorMatDual& obj) {
        os << "TensorMatDual Object:" << "\n";
        os << "  Real Part (r): " << obj.r << "\n";
        os << "    Shape: " << obj.r.sizes() << "\n";
        os << "  Dual Part (d): " << obj.d << "\n";
        os << "    Shape: " << obj.d.sizes() << "\n";
        return os;
    }

    /**
     * @brief Squeezes all dimensions of size 1 in the real and dual parts of the TensorDual.
     *
     * This method removes all dimensions of size 1 in the real (`r`) and dual (`d`) tensors.
     * If specific dimensions need to be squeezed, consider adding a parameter to specify
     * the target dimension(s) instead of using this method.
     *
     * @return TensorDual A new TensorDual object with squeezed real and dual parts.
     * @throws std::runtime_error If the tensors are not defined or have invalid shapes.
     */
    TensorDual squeeze() const {

        // Squeeze all dimensions of size 1 for both tensors
        auto r_squeezed = this->r.squeeze();
        auto d_squeezed = this->d.squeeze();

        // Return the new TensorDual object
        return TensorDual(r_squeezed, d_squeezed);
    }




    /**
     * @brief Squeezes the specified dimension of the real and dual tensors.
     *
     * This method removes the specified dimension from the real (`r`) and dual (`d`)
     * tensors if its size is 1. The resulting tensors are used to construct and return
     * a new `TensorDual` object.
     *
     * @param dim The dimension to squeeze.
     * @return TensorDual A new `TensorDual` object with the specified dimension squeezed.
     * @throws std::invalid_argument If the specified dimension is invalid or cannot be squeezed.
     */
    TensorDual squeeze(int dim) const {
        // Validate the dimension for the real tensor
        if (dim < 0 || dim >= this->r.dim()) {
            throw std::invalid_argument("Invalid dimension for squeezing the real tensor");
        }
        if (this->r.size(dim) != 1) {
            throw std::invalid_argument("Dimension size is not 1 for the real tensor, cannot squeeze");
        }

        // Validate the dimension for the dual tensor
        if (dim < 0 || dim >= this->d.dim()) {
            throw std::invalid_argument("Invalid dimension for squeezing the dual tensor");
        }
        if (this->d.size(dim) != 1) {
            throw std::invalid_argument("Dimension size is not 1 for the dual tensor, cannot squeeze");
        }

        // Squeeze the tensors and return a new TensorDual
        auto r = this->r.squeeze(dim);
        auto d = this->d.squeeze(dim);
        return TensorDual(r, d);
    }


    /**
     * @brief Ensures the real and dual tensors are in a contiguous memory layout.
     *
     * This method applies the `contiguous()` operation to both the real (`r`) and dual (`d`) tensors,
     * ensuring they are stored in a contiguous memory layout. A new `TensorMatDual` object is returned
     * with the resulting tensors.
     *
     * @return TensorMatDual A new `TensorMatDual` object with contiguous real and dual tensors.
     */
    TensorMatDual contiguous() const {
        // Apply contiguous() to real and dual parts
        auto r = this->r.contiguous();
        auto d = this->d.contiguous();
        // Return a new TensorMatDual object
        return TensorMatDual(r, d);
    }

    /**
     * @brief Generates a TensorMatDual object with the real part as a batch of identity matrices
     * and the dual part as a zero tensor.
     *
     * The `eye()` method constructs:
     * - The real part (`r`): A batch of identity matrices with the same batch size as the original `r`.
     * - The dual part (`d`): A zero tensor with the same batch size and dual dimensions as the original `d`.
     *
     * @return TensorMatDual A new TensorMatDual object with identity matrices and zero tensors.
     * @throws std::runtime_error If the input tensor shapes are invalid.
     */
    TensorMatDual eye() const {
        // Validate dimensions for `r` and `d`
        if (this->r.dim() < 2) {
            throw std::runtime_error("Real tensor `r` must have at least 2 dimensions");
        }
        if (this->d.dim() < 4) {
            throw std::runtime_error("Dual tensor `d` must have at least 4 dimensions");
        }

        // Create a batch of identity matrices for `r`
        auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});

        // Create a zero tensor for `d` with the same batch size and dimensions
        auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(2), this->d.size(3)}, this->d.options());

        // Return the resulting TensorMatDual
        return TensorMatDual(r, d);
    }


    /**
     * @brief Computes the sum of the real and dual tensors along a specified dimension.
     *
     * This method calculates the sum of the real (`r`) and dual (`d`) tensors along the
     * given dimension, retaining the reduced dimension in the output tensors.
     *
     * @param dim The dimension along which to compute the sum.
     * @return TensorMatDual A new TensorMatDual object containing the summed real and dual tensors.
     * @throws std::invalid_argument If the specified dimension is invalid for the tensors.
     */
    TensorMatDual sum(int dim) const {
        // Validate the dimension for `r` and `d`
        if (dim < 0 || dim >= this->r.dim()) {
            throw std::invalid_argument("Invalid dimension for summing the real tensor");
        }
        if (dim < 0 || dim >= this->d.dim()) {
            throw std::invalid_argument("Invalid dimension for summing the dual tensor");
        }

        // Compute the sum along the specified dimension
        auto r = this->r.sum(dim, true);
        auto d = this->d.sum(dim, true);

        // Return a new TensorMatDual object
        return TensorMatDual(r, d);
    }

    /**
     * @brief Computes the square of the TensorMatDual object.
     *
     * This method calculates the square of the real part (`r`) and the corresponding dual part
     * using the derivative of the squaring operation. Specifically:
     * - Real part (`r`): \( r^2 \)
     * - Dual part (`d`): \( 2r \cdot d \), where \( r \) is unsqueezed along the last dimension.
     *
     * @return TensorMatDual A new TensorMatDual object representing the squared values.
     * @throws std::runtime_error If the real and dual parts are not compatible for broadcasting.
     */
    TensorMatDual square() const {
        // Compute the square of the real part
        auto rsq = this->r.square();

        // Compute the dual part
        if (this->r.dim() + 1 != this->d.dim()) {
            throw std::runtime_error("Shape mismatch between real and dual tensors");
        }
        auto d = 2 * this->r.unsqueeze(-1) * this->d;

        // Return the squared TensorMatDual
        return TensorMatDual(rsq, d);
    }

    /**
     * @brief Computes the square root of the TensorMatDual object.
     *
     * This method calculates:
     * - The square root of the real part (`r`).
     * - The corresponding dual part using the chain rule for the square root operation.
     *
     * For negative values in the real part, the method throws an exception,
     * as the square root is undefined for such inputs in real-valued tensors.
     *
     * @return TensorMatDual A new TensorMatDual object with the square root of the real and dual parts.
     * @throws std::runtime_error If the real part contains negative values or if dimensions are incompatible.
     */
    TensorMatDual sqrt() const {
        TensorMatDual c;
        // Validate the real part
        if (torch::any(this->r < 0).item<bool>()) {
            //Convert to a complex number
            auto c = this->complex();
        }
        else {
            c = *this;
        }


        // Compute the square root of the real part
        auto r = torch::sqrt(c.r);


        // Compute the dual part using the chain rule: d = (1 / (2 * sqrt(r))) * d
        auto rf_inv_sqrt = 0.5 * c.r.pow(-0.5); // Add dimension for broadcasting
        auto d = torch::einsum("mij, mijn->mijn", {rf_inv_sqrt, c.d});

        // Return the resulting TensorMatDual
        return TensorMatDual(r, d);
    }

    /**
     * @brief Computes the L2 norm of the real part and its corresponding dual part.
     *
     * This method calculates:
     * - The L2 norm of the real part (`r`) along the last dimension.
     * - The corresponding dual part using the derivative of the L2 norm operation.
     *
     * @return TensorMatDual A new TensorMatDual object with the L2 norm as the real part
     * and the appropriately transformed dual part.
     * @throws std::runtime_error If the dimensions of the real and dual tensors are incompatible.
     */
    TensorMatDual normL2() const {
        // Compute the L2 norm of the real part along the last dimension
        auto norm_r = torch::norm(this->r, 2, -1, true); // Keep reduced dimension

        // Handle division by zero
        auto norm_r_expanded = norm_r.expand_as(this->r);
        auto grad_r = torch::where(
            norm_r_expanded > 0, 
            this->r / norm_r_expanded, 
            torch::zeros_like(this->r)
        );

        // Validate dimensions of `r` and `d`
        if (this->d.dim() != this->r.dim() + 1) {
            throw std::runtime_error("Shape mismatch: Dual part must have one more dimension than the real part.");
        }

        // Compute the dual part using the chain rule
        auto dual = torch::einsum("mij, mijn->min", {grad_r, this->d}).unsqueeze(2);

        // Return the resulting TensorMatDual
        return TensorMatDual(norm_r, dual);
    }
    /**
     * @brief Creates a TensorMatDual object with a given real tensor and a zero dual tensor.
     *
     * This static method constructs a `TensorMatDual` object:
     * - The real part (`r`) is set to the provided tensor.
     * - The dual part is a zero tensor with the same shape as `r` but with an additional
     *   dimension of size `ddim` appended to the end.
     *
     * @param r The real part tensor.
     * @param ddim The size of the additional dimension for the dual part.
     * @return TensorMatDual A new TensorMatDual object with the specified real tensor and zero dual tensor.
     * @throws std::invalid_argument If `ddim` is negative or `r` is not a valid tensor.
     */
    static TensorMatDual createZero(const torch::Tensor& r, int ddim) {
        // Validate the real tensor
        if (!r.defined()) {
            throw std::invalid_argument("The input real tensor `r` must be defined.");
        }

        // Validate the dual dimension
        if (ddim <= 0) {
            throw std::invalid_argument("The dual dimension `ddim` must be positive.");
        }

        // Compute the shape for the dual tensor
        auto dshape = r.sizes().vec(); // Copy the sizes to a vector
        dshape.push_back(ddim);        // Add the extra dimension for the dual part

        // Create a zero tensor for the dual part
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);

        // Return the resulting TensorMatDual object
        return TensorMatDual(r, ds);
    }



    /**
     * @brief Creates a TensorMatDual object with zero tensors, matching the shapes, dtypes, and devices of the current object.
     *
     * This method generates:
     * - A zero tensor for the real part (`r`) with the same shape, dtype, and device as the current real part.
     * - A zero tensor for the dual part (`d`) with the same shape, dtype, and device as the current dual part.
     *
     * @return TensorMatDual A new TensorMatDual object with zero-filled tensors for the real and dual parts.
     * @throws std::runtime_error If the real or dual tensors of the current object are not defined.
     */
    TensorMatDual zeros_like() const {
        // Validate that the real and dual tensors are defined

        // Create zero tensors for the real and dual parts
        auto rc = torch::zeros_like(this->r);
        auto dc = torch::zeros_like(this->d);

        // Return the new TensorMatDual object
        return TensorMatDual(rc, dc);
    }

    /**
     * @brief Creates a deep copy of the current TensorMatDual object.
     *
     * This method clones both the real (`r`) and dual (`d`) tensors to create
     * a new TensorMatDual object that is independent of the original.
     *
     * - `torch::clone()` creates a deep copy of the tensor, ensuring that
     *   changes to the cloned tensors do not affect the original tensors.
     *
     * @return TensorMatDual A new TensorMatDual object with cloned real and dual parts.
     * @throws std::runtime_error If the real or dual tensors are not defined.
     */
    TensorMatDual clone() const {

        // Clone the real and dual tensors
        return TensorMatDual(this->r.clone(), this->d.clone());
    }



    /**
     * @brief Concatenates two TensorMatDual objects along a specified dimension.
     *
     * This method concatenates the real (`r`) and dual (`d`) tensors of two TensorMatDual objects
     * along a specified dimension. By default, the concatenation occurs along dimension 2.
     *
     * @param t1 The first TensorMatDual object.
     * @param t2 The second TensorMatDual object.
     * @param dim The dimension along which to concatenate the tensors (default is 2).
     * @return TensorMatDual A new TensorMatDual object with concatenated real and dual parts.
     * @throws std::invalid_argument If the tensors are not compatible for concatenation.
     */
    static TensorMatDual cat(const TensorMatDual& t1, const TensorMatDual& t2, int dim = 2) {
        // Validate that the tensors are defined
        if (!t1.r.defined() || !t2.r.defined() || !t1.d.defined() || !t2.d.defined()) {
            throw std::invalid_argument("Cannot concatenate TensorMatDual objects: undefined tensors.");
        }

        // Validate shapes for real tensors
        if (t1.r.sizes().size() != t2.r.sizes().size()) {
            throw std::invalid_argument("Real tensors must have the same number of dimensions.");
        }
        for (int i = 0; i < t1.r.sizes().size(); ++i) {
            if (i != dim && t1.r.size(i) != t2.r.size(i)) {
                throw std::invalid_argument("Real tensors must match in all dimensions except the concatenation dimension.");
            }
        }

        // Validate shapes for dual tensors
        if (t1.d.sizes().size() != t2.d.sizes().size()) {
            throw std::invalid_argument("Dual tensors must have the same number of dimensions.");
        }
        for (int i = 0; i < t1.d.sizes().size(); ++i) {
            if (i != dim && t1.d.size(i) != t2.d.size(i)) {
                throw std::invalid_argument("Dual tensors must match in all dimensions except the concatenation dimension.");
            }
        }

        // Concatenate real and dual parts along the specified dimension
        auto r = torch::cat({t1.r, t2.r}, dim);
        auto d = torch::cat({t1.d, t2.d}, dim);

        // Return the new TensorMatDual object
        return TensorMatDual(r, d);
    }
    
    
    /**
     * @brief Concatenates a TensorMatDual object with a TensorDual object along the third dimension.
     *
     * This method concatenates the `r` (real) and `d` (dual) tensors of a `TensorMatDual` object (`t1`)
     * with a `TensorDual` object (`t2`) along the third dimension. The `TensorDual` tensors are unsqueezed
     * to match the required dimensionality before concatenation.
     *
     * @param t1 The TensorMatDual object.
     * @param t2 The TensorDual object.
     * @return TensorMatDual A new TensorMatDual object with concatenated real and dual parts.
     * @throws std::invalid_argument If the tensors are incompatible for concatenation.
     */
    static TensorMatDual cat(const TensorMatDual& t1, const TensorDual& t2) {

        // Validate the batch size and other dimensions for real tensors
        if (t1.r.size(0) != t2.r.size(0)) {
            throw std::invalid_argument("Batch sizes must match for concatenation.");
        }
        if (t1.r.size(1) != t2.r.size(1)) {
            throw std::invalid_argument("Real tensors must match in dimension 1.");
        }

        // Validate the batch size and other dimensions for dual tensors
        if (t1.d.size(0) != t2.d.size(0)) {
            throw std::invalid_argument("Batch sizes must match for concatenation of dual tensors.");
        }
        if (t1.d.size(1) != t2.d.size(1)) {
            throw std::invalid_argument("Dual tensors must match in dimension 1.");
        }
        if (t1.d.size(3) != t2.d.size(2)) {
            throw std::invalid_argument("Dual tensors must match in their last dimension after unsqueeze.");
        }

        // Unsqueeze and concatenate the real and dual parts
        auto r = torch::cat({t1.r, t2.r.unsqueeze(2)}, 2);
        auto d = torch::cat({t1.d, t2.d.unsqueeze(2)}, 2);

        // Return the concatenated TensorMatDual
        return TensorMatDual(r, d);
    }


    /**
     * @brief Concatenates a TensorMatDual object with a single Tensor along a specified dimension.
     *
     * This method concatenates the real part (`r`) of a `TensorMatDual` object (`t1`) with a given tensor (`t2`).
     * The `t2` tensor is repeated to match the batch size of `t1.r`. The corresponding dual part is set to zeros.
     *
     * @param t1 The TensorMatDual object.
     * @param t2 The Tensor to concatenate with the real part of `t1`.
     * @param dim The dimension along which to concatenate the tensors (default is 2).
     * @return TensorMatDual A new TensorMatDual object with concatenated real and zero dual parts.
     * @throws std::invalid_argument If the tensors are incompatible for concatenation.
     */
    static TensorMatDual cat(const TensorMatDual& t1, const torch::Tensor& t2, int dim = 2) {

        // Validate dimensions of `t1.r` and `t2`
        if (t1.r.dim() != t2.dim() + 1) {
            throw std::invalid_argument("The input tensor `t2` must have one less dimension than `t1.r`.");
        }
        if (t1.r.size(1) != t2.size(0)) {
            throw std::invalid_argument("The second dimension of `t1.r` must match the first dimension of `t2`.");
        }

        // Repeat `t2` to match the batch size of `t1.r`
        auto rt = t2.repeat({t1.r.size(0), 1, 1});

        // Concatenate the real part
        auto r = torch::cat({t1.r, rt}, dim);

        // Create a zero tensor for the new dual part corresponding to `t2`
        auto d_shape = t1.d.sizes().vec();
        d_shape[dim] = rt.size(dim); // Adjust the concatenation dimension size
        auto zeros_for_t2 = torch::zeros(d_shape, t1.d.options());

        // Concatenate the dual part
        auto d = torch::cat({t1.d, zeros_for_t2}, dim);

        // Return the concatenated TensorMatDual
        return TensorMatDual(r, d);
    }


    /**
     * @brief Overloads the + operator to add two TensorMatDual objects.
     *
     * This operator adds the corresponding `r` (real) and `d` (dual) tensors of two
     * `TensorMatDual` objects element-wise. The two objects must have compatible shapes.
     *
     * @param other The TensorMatDual object to add to this object.
     * @return TensorMatDual A new TensorMatDual object representing the element-wise sum.
     * @throws std::invalid_argument If the real or dual tensors are incompatible for addition.
     */
    TensorMatDual operator+(const TensorMatDual& other) const {
        // Validate that the tensors are defined
        if (!this->r.defined() || !this->d.defined() || !other.r.defined() || !other.d.defined()) {
            throw std::invalid_argument("Cannot add TensorMatDual objects: undefined tensors.");
        }

        // Validate that the shapes match
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Cannot add TensorMatDual objects: incompatible tensor shapes.");
        }

        // Perform element-wise addition
        return TensorMatDual(this->r + other.r, this->d + other.d);
    }


    /**
     * @brief Overloads the + operator to add a TensorMatDual object with a TensorDual object.
     *
     * This operator adds the real (`r`) and dual (`d`) tensors of a `TensorDual` object to
     * the corresponding tensors in a `TensorMatDual` object element-wise. The tensors must
     * have compatible shapes.
     *
     * @param other The TensorDual object to add to this TensorMatDual object.
     * @return TensorMatDual A new TensorMatDual object representing the element-wise sum.
     * @throws std::invalid_argument If the tensors are undefined or incompatible for addition.
     */
    TensorMatDual operator+(const TensorDual& other) const {


        // Perform element-wise addition
        auto rn = torch::einsum("mij,mi->mij", {this->r, other.r});
        auto dn = torch::einsum("mijk, mik->mijk", {this->d, other.d});
        return TensorMatDual(rn, dn);
    }

    /**
     * @brief Overloads the + operator to add a scalar (double) to the real part of a TensorMatDual object.
     *
     * This operator adds the scalar `other` to the real (`r`) part of the `TensorMatDual` object,
     * leaving the dual (`d`) part unchanged.
     *
     * @param other The scalar (double) to add to the real part.
     * @return TensorMatDual A new TensorMatDual object with the scalar added to the real part.
     * @throws std::runtime_error If the real tensor (`r`) is not defined.
     */
    TensorMatDual operator+(const double& other) const {

        // Add the scalar to the real part and return a new TensorMatDual
        return TensorMatDual(this->r + other, this->d);
    }

    /**
     * @brief Overloads the - operator to subtract one TensorMatDual object from another.
     *
     * This operator subtracts the real (`r`) and dual (`d`) tensors of one `TensorMatDual` object
     * (`other`) from the corresponding tensors in this object, element-wise. The two objects must
     * have compatible shapes.
     *
     * @param other The TensorMatDual object to subtract from this object.
     * @return TensorMatDual A new TensorMatDual object representing the element-wise difference.
     * @throws std::invalid_argument If the tensors are undefined or incompatible for subtraction.
     */
    TensorMatDual operator-(const TensorMatDual& other) const {

        // Perform element-wise subtraction
        return TensorMatDual(this->r - other.r, this->d - other.d);
    }


    /**
     * @brief Overloads the - operator to subtract a scalar (double) from the real part of a TensorMatDual object.
     *
     * This operator subtracts the scalar `other` from the real (`r`) part of the `TensorMatDual` object,
     * leaving the dual (`d`) part unchanged.
     *
     * @param other The scalar (double) to subtract from the real part.
     * @return TensorMatDual A new TensorMatDual object with the scalar subtracted from the real part.
     */
    TensorMatDual operator-(const double& other) const {

        // Subtract the scalar from the real part and return a new TensorMatDual
        return TensorMatDual(this->r - other, this->d);
    }


    /**
     * @brief Overloads the == operator to compare two TensorMatDual objects element-wise.
     *
     * This operator compares the real (`r`) parts of two TensorMatDual objects element-wise
     * and returns a mask tensor indicating where the elements are equal. The shapes of the
     * real parts (`r`) must be compatible for comparison.
     *
     * @param other The TensorMatDual object to compare with this object.
     * @return torch::Tensor A boolean tensor (mask) of the same shape as the real part (`r`),
     * indicating where the elements of the two objects are equal.
     * @throws std::invalid_argument If the real tensors (`r`) are undefined or incompatible for comparison.
     */
    torch::Tensor operator==(const TensorMatDual& other) const {

        // Validate that the shapes match
        if (this->r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Cannot compare TensorMatDual objects: incompatible tensor shapes.");
        }

        // Perform element-wise equality comparison for the real parts
        return this->r == other.r;
    }


    /**
     * @brief Overloads the unary - operator to negate the real and dual parts of a TensorMatDual object.
     *
     * This operator returns a new TensorMatDual object where both the real (`r`) and dual (`d`) tensors
     * are negated element-wise.
     *
     * @return TensorMatDual A new TensorMatDual object with the real and dual parts negated.
     */
    TensorMatDual operator-() const {

        // Negate the real and dual parts and return a new TensorMatDual
        return TensorMatDual(-this->r, -this->d);
    }

    /**
     * @brief Overloads the * operator to scale the real and dual parts of a TensorMatDual object by a scalar.
     *
     * This operator scales both the real (`r`) and dual (`d`) tensors of the TensorMatDual object
     * element-wise by the scalar `other`.
     *
     * @param other The scalar (double) by which to scale the real and dual tensors.
     * @return TensorMatDual A new TensorMatDual object with the real and dual parts scaled by the scalar.
     */
    TensorMatDual operator*(const double other) const {

        // Scale the real and dual parts
        auto real = this->r * other;
        auto dual = this->d * other;

        // Return a new TensorMatDual object
        return TensorMatDual(real, dual);
    }


    /**
     * @brief Overloads the / operator to perform element-wise division of two TensorMatDual objects.
     *
     * This operator divides the real (`r`) and dual (`d`) parts of the current `TensorMatDual` object
     * by the corresponding parts of another `TensorMatDual` object (`other`). The operation uses
     * the chain rule for differentiation to compute the dual part.
     *
     * @param other The TensorMatDual object to divide by.
     * @return TensorMatDual A new TensorMatDual object representing the result of the division.
     * @throws std::invalid_argument If the tensors are undefined or incompatible for division.
     * @throws std::runtime_error If division by zero occurs in the real part.
     */
    TensorMatDual operator/(const TensorMatDual& other) const {
        // Validate that the shapes match
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Cannot divide TensorMatDual objects: incompatible tensor shapes.");
        }

        // Check for division by zero in the real part
        if (torch::any(other.r == 0).item<bool>()) {
            throw std::runtime_error("Division by zero encountered in the real part.");
        }

        // Perform element-wise division for the real part
        auto r = this->r / other.r;

        // Compute the square of the real part of `other`
        auto otherrsq = other.r.square();

        // Compute the dual part using the chain rule
        auto d = -torch::einsum("mij,mijn->mijn", {this->r / otherrsq, other.d}) + 
                  torch::einsum("mij, mijn->mijn",{other.r.reciprocal(), this->d});

        // Return the result as a new TensorMatDual object
        return TensorMatDual(r, d);
    }



    /**
     * @brief Overloads the / operator to perform element-wise division of a TensorMatDual object by a torch::Tensor.
     *
     * This operator divides the real (`r`) and dual (`d`) tensors of the `TensorMatDual` object
     * element-wise by a `torch::Tensor`. The provided tensor (`other`) must be compatible in shape
     * for broadcasting with the real and dual tensors of the `TensorMatDual` object.
     *
     * @param other The torch::Tensor by which to divide the TensorMatDual object.
     * @return TensorMatDual A new TensorMatDual object with the real and dual parts divided by `other`.
     * @throws std::invalid_argument If the input tensor (`other`) is undefined or incompatible for division.
     * @throws std::runtime_error If division by zero occurs in the input tensor (`other`).
     */
    TensorMatDual operator/(const torch::Tensor& other) const {
        // Validate that the input tensor is defined
        if (!other.defined()) {
            throw std::invalid_argument("Cannot divide TensorMatDual by undefined tensor.");
        }

        // Validate that the shapes are compatible for broadcasting
        if (this->r.sizes() != other.sizes() && !other.sizes().empty()) {
            throw std::invalid_argument("Cannot divide TensorMatDual by tensor: incompatible shapes for broadcasting.");
        }

        // Check for division by zero in the input tensor
        if (torch::any(other == 0).item<bool>()) {
            throw std::runtime_error("Division by zero encountered in the input tensor.");
        }

        // Perform element-wise division for the real and dual parts
        auto real = this->r / other;
        auto dual = this->d / other.unsqueeze(-1);

        // Return a new TensorMatDual object
        return TensorMatDual(real, dual);
    }
    
    /**
     * @brief Overloads the / operator to scale the real and dual parts of a TensorMatDual object by the reciprocal of a scalar.
     *
     * This operator divides both the real (`r`) and dual (`d`) tensors of the TensorMatDual object
     * element-wise by the scalar `other`.
     *
     * @param other The scalar (double) by which to divide the real and dual tensors.
     * @return TensorMatDual A new TensorMatDual object with the real and dual parts scaled by the reciprocal of the scalar.
     * @throws std::runtime_error If the scalar value is zero.
     */
    TensorMatDual operator/(const double other) const {
        // Check for division by zero
        if (other == 0) {
            throw std::runtime_error("Division by zero encountered in scalar division.");
        }

        // Scale the real and dual parts
        auto real = this->r / other;
        auto dual = this->d / other;

        // Return a new TensorMatDual object
        return TensorMatDual(real, dual);
    }

    /**
     * @brief Indexes the real and dual parts of a TensorMatDual object.
     *
     * This method applies the specified indices to both the real (`r`) and dual (`d`) tensors
     * of the TensorMatDual object. If the resulting tensors have fewer dimensions than expected
     * (2 for `r` or 3 for `d`), an additional dimension is added to maintain compatibility.
     *
     * @param indices A vector of torch::indexing::TensorIndex objects specifying the indices.
     * @return TensorMatDual A new TensorMatDual object with indexed real and dual tensors.
     * @throws std::invalid_argument If the real or dual tensors are undefined.
     */
    TensorMatDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {

        // Index the real part
        auto r = this->r.index(indices);


        // Index the dual part
        auto d = this->d.index(indices);


        // Return the indexed TensorMatDual object
        return TensorMatDual(r, d);
    }



    /**
     * @brief Indexes the real and dual parts of a TensorMatDual object along the first dimension.
     *
     * This method extracts a specific slice of the real (`r`) and dual (`d`) tensors
     * from the first dimension, corresponding to the provided index.
     *
     * @param index The index specifying the slice to extract.
     * @return TensorMatDual A new TensorMatDual object containing the indexed slice.
     * @throws std::out_of_range If the index is out of bounds.
     * @throws std::invalid_argument If the real or dual tensors are undefined.
     */
    TensorMatDual index(int index) const {
        //Check if the index is greater than the first dimension and trow an error
        if (index >= this->r.size(0)) {
            throw std::out_of_range("Index out of bounds for indexing the first dimension of the TensorMatDual object.");
        }

        //Convert this into a slice to avoid reduction of dimensions
        // Perform indexing
        auto real = this->r.index({index}).unsqueeze(0);
        auto dual = this->d.index({index}).unsqueeze(0);

        // Return the indexed TensorMatDual object
        return TensorMatDual(real, dual);
    }


    /**
     * @brief Indexes the real and dual parts of a TensorMatDual object using a mask tensor.
     *
     * This method applies the specified mask to both the real (`r`) and dual (`d`) tensors of the
     * TensorMatDual object. The mask must be a boolean tensor of the same shape as the first dimension
     * of `r` and `d`, or it must be broadcastable to those dimensions.
     *
     * @param mask A boolean torch::Tensor specifying the elements to index.
     * @return TensorMatDual A new TensorMatDual object containing the indexed elements.
     * @throws std::invalid_argument If the mask is undefined or incompatible for indexing.
     */
    TensorMatDual index(const torch::Tensor& mask) const {

        // Validate that the mask is defined
        if (!mask.defined()) {
            throw std::invalid_argument("Cannot index TensorMatDual: undefined mask tensor.");
        }

        // Validate that the mask is a boolean tensor
        if (mask.dtype() != torch::kBool) {
            throw std::invalid_argument("Cannot index TensorMatDual: mask tensor must be of boolean type.");
        }


        // Apply the mask to the real and dual parts
        auto real = this->r.index({mask});
        auto dual = this->d.index({mask});

        // Return the indexed TensorMatDual object
        return TensorMatDual(real, dual);
    }



    /**
     * @brief Sets the requires_grad attribute for the real and dual tensors in a TensorMatDual object.
     *
     * This method sets the `requires_grad` attribute for both the real (`r`) and dual (`d`) tensors
     * of the TensorMatDual object. This attribute is used to track gradients during autograd operations.
     *
     * @param req_grad A boolean value indicating whether to enable or disable gradient tracking.
     * @throws std::invalid_argument If the real or dual tensors are undefined.
     */
    void requires_grad_(bool req_grad) {

        // Set the requires_grad attribute for both tensors
        r.requires_grad_(req_grad);
        d.requires_grad_(req_grad);
    }


    /**
     * @brief Triggers the backward pass for the real and dual tensors in a TensorMatDual object.
     *
     * This method calls the `backward` function on both the real (`r`) and dual (`d`) tensors
     * to compute gradients. It assumes that both tensors have `requires_grad` set to `true`.
     *
     * @throws std::runtime_error If the real or dual tensors are undefined or do not require gradients.
     */
    void backward() {

        // Validate that the tensors require gradients
        if (!r.requires_grad() || !d.requires_grad()) {
            throw std::runtime_error("Cannot perform backward pass: real or dual tensors do not require gradients.");
        }

        // Trigger the backward pass for both tensors
        r.backward();
        d.backward();
    }



    /**
     * @brief Computes the element-wise absolute value of a TensorMatDual object.
     *
     * This method computes the absolute value of the real (`r`) tensor and adjusts the dual (`d`) tensor
     * accordingly. For complex tensors, it uses the real part of the `r` tensor to compute the sign.
     *
     * @return TensorMatDual A new TensorMatDual object with the absolute value of the real part
     * and the adjusted dual part.
     * @throws std::invalid_argument If the real or dual tensors are undefined.
     */
    TensorMatDual abs() const {

        // Compute the absolute value of the real part
        auto abs_r = torch::abs(r);

        // Compute the sign of the real part
        auto sign_r = torch::is_complex(r) ? torch::sign(torch::real(r)) : torch::sign(r);

        // Adjust the dual part
        auto abs_d = sign_r.unsqueeze(-1) * d;

        // Return the result as a new TensorMatDual object
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


    /**
     * @brief Static member function to perform einsum on a TensorMatDual and a torch::Tensor.
     *
     * This function computes the einsum operation for the real (`r`) and dual (`d`) parts of the 
     * TensorMatDual object and a torch::Tensor. The einsum string is used to define the computation.
     *
     * @param arg The einsum string defining the operation.
     * @param first The TensorMatDual object (first argument for einsum).
     * @param second The torch::Tensor (second argument for einsum).
     * @return TensorMatDual A new TensorMatDual object containing the computed real and dual tensors.
     * @throws std::invalid_argument If the einsum string is invalid or the input tensors are undefined.
     */
    static TensorMatDual einsum(const std::string& arg, const TensorMatDual& first, const torch::Tensor& second) {

        // Validate the einsum string format
        auto pos = arg.find(",");
        if (pos == std::string::npos || arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: missing ',' or '->'.");
        }

        // Compute the real part using the provided einsum string
        auto r = torch::einsum(arg, {first.r, second});

        // Parse the einsum string to modify it for dual computation
        int pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos);                 // The indices for the first input
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1); // The indices for the second input
        auto arg3 = arg.substr(pos2 + 2);               // The output indices

        // Adjust the einsum string for the dual computation
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        // Compute the dual part using the modified einsum string
        auto d1 = torch::einsum(darg1, {first.d, second});

        // Return the resulting TensorMatDual
        return TensorMatDual(std::move(r), std::move(d1));
    }


    /**
     * @brief Static member function to perform einsum on a TensorDual and a TensorMatDual object.
     *
     * This function computes the einsum operation for the real (`r`) and dual (`d`) parts of
     * a TensorDual and a TensorMatDual object. The einsum string defines the operation.
     *
     * @param arg The einsum string defining the operation.
     * @param first The TensorDual object (first argument for einsum).
     * @param second The TensorMatDual object (second argument for einsum).
     * @return TensorMatDual A new TensorMatDual object containing the computed real and dual tensors.
     * @throws std::invalid_argument If the einsum string is invalid or the input tensors are undefined.
     */
    static TensorDual einsum(const std::string& arg, const TensorDual& first, const TensorMatDual& second) {
        // Validate that the inputs are defined
        if (!first.r.defined() || !first.d.defined() || !second.r.defined() || !second.d.defined()) {
            throw std::invalid_argument("Cannot perform einsum: undefined real or dual tensors in inputs.");
        }

        // Validate the einsum string format
        auto pos = arg.find(",");
        if (pos == std::string::npos || arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: missing ',' or '->'.");
        }

        // Parse the einsum string
        int pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos);                 // The indices for the first input
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1); // The indices for the second input
        auto arg3 = arg.substr(pos2 + 2);               // The output indices

        // Compute the real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Adjust the einsum strings for the dual computation
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1 = torch::einsum(darg1, {first.d, second.r});

        auto darg2 = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        // Combine the dual computations
        auto d = d1 + d2;

        // Return the resulting TensorMatDual object
        return TensorDual(std::move(r), std::move(d));
    }

    /**
     * @brief Static member function to perform einsum on a torch::Tensor and a TensorMatDual object.
     *
     * This function computes the einsum operation for the real (`r`) and dual (`d`) parts of
     * a torch::Tensor and a TensorMatDual object. The einsum string defines the operation.
     *
     * @param arg The einsum string defining the operation.
     * @param first The torch::Tensor (first argument for einsum).
     * @param second The TensorMatDual object (second argument for einsum).
     * @return TensorMatDual A new TensorMatDual object containing the computed real and dual tensors.
     * @throws std::invalid_argument If the einsum string is invalid or the input tensors are undefined.
     */
    static TensorMatDual einsum(const std::string& arg, const torch::Tensor& first, const TensorMatDual& second) {
        // Validate that the inputs are defined
        if (!first.defined() || !second.r.defined() || !second.d.defined()) {
            throw std::invalid_argument("Cannot perform einsum: undefined real or dual tensors in inputs.");
        }

        // Validate the einsum string format
        auto pos = arg.find(",");
        if (pos == std::string::npos || arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: missing ',' or '->'.");
        }

        // Parse the einsum string
        int pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos);                 // The indices for the first input
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1); // The indices for the second input
        auto arg3 = arg.substr(pos2 + 2);               // The output indices

        // Compute the real part
        auto r = torch::einsum(arg, {first, second.r});

        // Adjust the einsum string for the dual computation
        auto darg1 = arg1 + "," + arg2 + "z->" + arg3 + "z";

        // Compute the dual part
        auto d = torch::einsum(darg1, {first, second.d});

        // Return the resulting TensorMatDual object
        return TensorMatDual(std::move(r), std::move(d));
    }


    /**
     * @brief Static member function to perform einsum on two TensorMatDual objects.
     *
     * This function computes the einsum operation for the real (`r`) and dual (`d`) parts of
     * two TensorMatDual objects. The einsum string defines the operation and specifies how 
     * the tensors should interact.
     *
     * @param arg The einsum string defining the operation.
     * @param first The first TensorMatDual object.
     * @param second The second TensorMatDual object.
     * @return TensorMatDual A new TensorMatDual object containing the computed real and dual tensors.
     * @throws std::invalid_argument If the einsum string is invalid or the input tensors are undefined.
     */
    static TensorMatDual einsum(const std::string& arg, const TensorMatDual& first, const TensorMatDual& second) {

        // Validate the einsum string format
        auto pos = arg.find(",");
        if (pos == std::string::npos || arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: missing ',' or '->'.");
        }

        // Parse the einsum string
        int pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos);                 // Indices for the first input
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1); // Indices for the second input
        auto arg3 = arg.substr(pos2 + 2);               // Output indices

        // Compute the real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Adjust the einsum string for dual computation
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1 = torch::einsum(darg1, {first.d, second.r});

        auto darg2 = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        // Combine the dual computations
        auto d = d1 + d2;

        // Return the resulting TensorMatDual
        return TensorMatDual(std::move(r), std::move(d));
    }


    /**
     * @brief Computes the maximum values and their corresponding dual values along a specified dimension.
     *
     * This method computes the maximum values of the real tensor (`r`) along a specified dimension,
     * and gathers the corresponding dual values (`d`) based on the indices of the maximum values.
     *
     * @param dim The dimension along which to compute the maximum. Default is 1.
     * @return TensorMatDual A new TensorMatDual object containing the maximum real values and
     * their corresponding dual values.
     * @throws std::invalid_argument If the real or dual tensors are undefined or the dimension is invalid.
     */
    TensorMatDual max(int dim = 1) {
        // Validate that the tensors are defined
        if (!r.defined() || !d.defined()) {
            throw std::invalid_argument("Cannot compute max: undefined real or dual tensors.");
        }

        // Ensure the specified dimension is valid
        if (dim < 0 || dim >= r.dim()) {
            throw std::invalid_argument("Invalid dimension specified for max operation.");
        }

        // Compute the maximum values and their indices
        auto max_result = torch::is_complex(r)
                            ? torch::max(torch::real(r), /*dim=*/dim, /*keepdim=*/true)
                            : torch::max(r, /*dim=*/dim, /*keepdim=*/true);
        auto max_indices = std::get<1>(max_result);

        // Expand the indices for the dual tensor
        auto d_indices = max_indices.unsqueeze(-1).expand({-1, -1, -1, d.size(-1)});

        // Gather the maximum real values and corresponding dual values
        auto real_values = std::get<0>(max_result);
        auto dual_values = torch::gather(d, dim, d_indices);

        // Return the resulting TensorMatDual object
        return TensorMatDual(real_values, dual_values);
    }

    /**
     * @brief Computes the minimum values and their corresponding dual values along a specified dimension.
     *
     * This method computes the minimum values of the real tensor (`r`) along a specified dimension,
     * and gathers the corresponding dual values (`d`) based on the indices of the minimum values.
     *
     * @param dim The dimension along which to compute the minimum. Default is 1.
     * @return TensorMatDual A new TensorMatDual object containing the minimum real values and
     * their corresponding dual values.
     * @throws std::invalid_argument If the real or dual tensors are undefined or the dimension is invalid.
     */
    TensorMatDual min(int dim = 1) {

        // Ensure the specified dimension is valid
        if (dim < 0 || dim >= r.dim()) {
            throw std::invalid_argument("Invalid dimension specified for min operation.");
        }

        // Compute the minimum values and their indices
        auto min_result = torch::is_complex(r)
                            ? torch::min(torch::real(r), /*dim=*/dim, /*keepdim=*/true)
                            : torch::min(r, /*dim=*/dim, /*keepdim=*/true);
        auto min_indices = std::get<1>(min_result);

        // Expand the indices for the dual tensor
        auto d_indices = min_indices.unsqueeze(-1).expand({-1, -1, -1, d.size(-1)});

        // Gather the minimum real values and corresponding dual values
        auto real_values = std::get<0>(min_result);
        auto dual_values = torch::gather(d, dim, d_indices);

        // Return the resulting TensorMatDual object
        return TensorMatDual(real_values, dual_values);
    }



    /**
     * @brief Updates elements in the TensorMatDual object based on a mask and a TensorMatDual value.
     *
     * This method performs an in-place update of the real (`r`) and dual (`d`) tensors in the
     * TensorMatDual object. The elements specified by the mask are updated with the corresponding
     * elements from the TensorDual value.
     *
     * @param mask A boolean tensor specifying the elements to update.
     * @param value A TensorDual object containing the new real and dual values to assign.
     * @throws std::invalid_argument If the mask or value tensors are undefined or have incompatible shapes.
     */
    void index_put_(const torch::Tensor& mask, const TensorMatDual& value) {
        // Validate that the mask and value tensors are defined
        if (!mask.defined()) {
            throw std::invalid_argument("Cannot perform index_put_: mask tensor is undefined.");
        }

        // Validate that the mask is a boolean tensor
        if (mask.dtype() != torch::kBool) {
            throw std::invalid_argument("Cannot perform index_put_: mask tensor must be of boolean type.");
        }


        // Perform the in-place update
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
    }


    /**
     * @brief Updates elements in the TensorMatDual object based on a vector of TensorIndex objects and a TensorDual value.
     *
     * This method performs an in-place update of the real (`r`) and dual (`d`) tensors in the
     * TensorMatDual object. The elements specified by the vector of TensorIndex objects are updated with
     * the corresponding elements from the TensorDual value.
     *
     * @param mask A vector of TensorIndex objects specifying the indices to update.
     * @param value A TensorDual object containing the new real and dual values to assign.
     * @throws std::invalid_argument If the value tensors are undefined or the mask is invalid.
     */
    void index_put_(const std::vector<TensorIndex>& mask, const TensorMatDual& value) {
        this->r.index_put_(mask, value.r);
        this->d.index_put_(mask, value.d);
    }




    /**
     * @brief Updates elements in the TensorMatDual object based on a vector of TensorIndex objects and a torch::Tensor value.
     *
     * This method performs an in-place update of the real (`r`) tensor in the TensorMatDual object
     * using the specified torch::Tensor value. The corresponding dual (`d`) tensor elements are set to zero.
     *
     * @param mask A vector of TensorIndex objects specifying the indices to update.
     * @param value A torch::Tensor to assign to the specified elements of the real tensor.
     * @throws std::invalid_argument If the value tensor is undefined, incompatible with the mask, or if the mask is invalid.
     */
    void index_put_(const std::vector<TensorIndex>& mask, const torch::Tensor& value) {

        // Validate compatibility of the value tensor with the mask
        if (value.sizes() != this->r.index({mask}).sizes()) {
            throw std::invalid_argument("Cannot perform index_put_: value tensor dimensions are incompatible with the specified mask.");
        }

        // Perform the in-place update for the real and dual tensors
        this->r.index_put_({mask}, value);
        auto sizes = this->r.sizes();
        auto dual_values = torch::zeros({sizes[0], sizes[1], sizes[2], this->d.size(-1)}, this->d.options());
        this->d.index_put_({mask}, dual_values);
    }
    


    /**
     * @brief Adds a singleton dimension to a TensorDual and returns it as a TensorMatDual.
     *
     * This method takes a `TensorDual` object and adds a singleton dimension at the specified
     * position for both its real (`r`) and dual (`d`) tensors. The resulting tensors are
     * encapsulated in a `TensorMatDual` object.
     *
     * @param x The TensorDual object to unsqueeze.
     * @param dim The dimension along which to add the singleton dimension.
     * @return TensorMatDual A new TensorMatDual object with the added singleton dimension.
     * @throws std::invalid_argument If the input tensors are undefined or the dimension is invalid.
     */
    static TensorMatDual unsqueeze(const TensorDual& x, int dim) {


        // Perform the unsqueeze operation for both real and dual parts
        auto r = x.r.unsqueeze(dim);
        auto d = x.d.unsqueeze(dim);

        // Return the resulting TensorMatDual
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
    
    // toString method
    std::string toString() const {
        std::ostringstream oss;
        oss << "TensorMatHyperDual("
            << "r: " << r.sizes() << ", "
            << "d: " << d.sizes() << ", "
            << "h: " << h.sizes() << ")";
        return oss.str();
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

    TensorMatHyperDual(const TensorDual& x, int dim = 2) : device_(x.r.device()) {
        if (dim <= 0 || dim > 2) {
            throw std::invalid_argument("Invalid dimension specified for TensorDual to TensorMatHyperDual conversion.");
        }
        dtype_ = torch::typeMetaToScalarType(x.r.dtype());

        auto dual_dim = x.d.sizes()[2];
        torch::Tensor rn, dn, hn;
        //Here h will have dimension [M, N, 1, D, D] where D is the dual dimension
        if (dim == 1) {
            this->r = x.r.unsqueeze(1);
            this->d = x.d.unsqueeze(1);
            std::vector<int64_t> h_sizes = {x.r.size(0), 1, x.r.size(1), dual_dim, dual_dim};
            this->h = torch::zeros(h_sizes, x.r.options());
        } else {
            this->r = x.r.unsqueeze(2);
            this->d = x.d.unsqueeze(2);
            std::vector<int64_t> h_sizes = {x.r.size(0), x.r.size(1), 1, dual_dim, dual_dim};
            this->h = torch::zeros(h_sizes, x.r.options());
        }
        
    }

    
    TensorMatHyperDual complex() const {
        torch::Tensor rc, dc, hc;

        // Convert real tensors to complex tensors
        if (this->r.is_complex()) {
            rc = this->r;
        } else {
            rc = torch::complex(this->r, torch::zeros_like(this->r));
        }

        if (this->d.is_complex()) {
            dc = this->d;
        } else {
            dc = torch::complex(this->d, torch::zeros_like(this->d));
        }

        if (this->h.is_complex()) {
            hc = this->h;
        } else {
            hc = torch::complex(this->h, torch::zeros_like(this->h));
        }

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
        auto szs = max_indices.sizes();
        int dsz = this->d.size(-1);
        int hsz = this->h.size(-1);
        

        auto dvalues = torch::gather(this->d, dim, max_indices.unsqueeze(-1).expand({szs[0], szs[1], szs[2], dsz}));

        auto hvalues = torch::gather(this->h, dim, max_indices.unsqueeze(-1).unsqueeze(-1).expand({szs[0], szs[1], szs[2], hsz, hsz}));

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(max_values, dvalues, hvalues);
    }

    /**
     * Compute the maximum values along a specified dimension for the TensorMatHyperDual object.
     *
     * The real, dual, and hyperdual parts are reduced based on the indices of the maximum values.
     *
     * @param dim The dimension along which to compute the maximum (default: 1).
     * @return A new TensorMatHyperDual object containing the maximum values and corresponding dual and hyperdual values.
     */
    TensorMatHyperDual min(int dim = 1) const {
        // Compute max values and indices along the specified dimension
        auto min_result = torch::min(this->r, dim, /*keepdim=*/true);
        auto min_values = std::get<0>(min_result);  // Maximum values
        auto min_indices = std::get<1>(min_result); // Indices of the maximum values
        auto szs = min_indices.sizes();
        int dsz = this->d.size(-1);
        int hsz = this->h.size(-1);
        

        auto dvalues = torch::gather(this->d, dim, min_indices.unsqueeze(-1).expand({szs[0], szs[1], szs[2], dsz}));

        auto hvalues = torch::gather(this->h, dim, min_indices.unsqueeze(-1).unsqueeze(-1).expand({szs[0], szs[1], szs[2], hsz, hsz}));

        // Return a new TensorMatHyperDual object
        return TensorMatHyperDual(min_values, dvalues, hvalues);
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
        if (dim == 0 || dim >= this->r.dim()) {
            throw std::invalid_argument("Invalid dimension specified for sum operation.");
        }
        if ( dim  < 0) {
            dim = this->r.dim()-1;
        }

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
        auto hn = 2 * torch::einsum("mijk, mijl->mijkl", {d, d}) + 
                2 * torch::einsum("mij, mijkl->mijkl", {r, h});

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

    //Forward declaration for eye function
    TensorMatHyperDual eye();

    /**
     * @brief Squeezes all dimensions of size 1 in the real and dual parts of the TensorDual.
     *
     * This method removes all dimensions of size 1 in the real (`r`) and dual (`d`) tensors.
     * If specific dimensions need to be squeezed, consider adding a parameter to specify
     * the target dimension(s) instead of using this method.
     *
     * @return TensorDual A new TensorDual object with squeezed real and dual parts.
     * @throws std::runtime_error If the tensors are not defined or have invalid shapes.
     */
    TensorHyperDual squeeze() const {

        // Squeeze all dimensions of size 1 for both tensors
        auto r_squeezed = this->r.squeeze();
        auto d_squeezed = this->d.squeeze();
        auto h_squeezed = this->h.squeeze();

        // Return the new TensorDual object
        return TensorHyperDual(r_squeezed, d_squeezed, h_squeezed);
    }



     
    /**
     * Squeeze the specified dimensio n of the TensorHyperDual object.
     *
     * Removes the specified dimension if it has size 1 from the real, dual, and hyperdual tensors.
     *
     * @param dim The dimension to squeeze. If the dimension does not have size 1, it remains unchanged.
     * @return A new TensorHyperDual object with the specified dimension squeezed.
     */
    TensorHyperDual squeeze(int dim) const {
        if ( dim  < 0) {
            dim = this->r.dim()-1;
        }
        if ( r.sizes()[dim] != 1) {
            throw std::invalid_argument("Cannot squeeze dimension with size greater than 1.");
        }
        if ( dim == 0 || dim >= this->r.dim()) {
            throw std::invalid_argument("Invalid dimension specified for squeeze operation.");
        }
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
        auto r_contiguous = this->r.is_contiguous() ? this->r : this->r.contiguous();
        auto d_contiguous = this->d.is_contiguous() ? this->d : this->d.contiguous();
        auto h_contiguous = this->h.is_contiguous() ? this->h : this->h.contiguous();
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
        TensorMatHyperDual tmp(*this);
        // Compute the square root of the real part
        if ((this->r < 0).any().item<bool>()) {
            tmp = tmp.complex();
        }
        auto r_sqrt = torch::sqrt(tmp.r);
        //replace zeros with small epsilon
        r_sqrt = torch::where(r_sqrt != 0, r_sqrt, torch::ones_like(r_sqrt) * 1e-12);

        // Compute the dual part: d / (2 * sqrt(r))
        auto rf_inv_sqrt = 0.5 * r_sqrt.pow(-1); // 1 / (2 * sqrt(r))
        auto d_sqrt = torch::einsum("mij, mijn->mijn", {rf_inv_sqrt, tmp.d});

        // Compute the hyperdual part: h / (2 * sqrt(r)) - d * d / (4 * r^(3/2))
        auto rf_inv_3_sqrt = 0.25 * r_sqrt.pow(-3); // 1 / (4 * r^(3/2))
        auto h_sqrt = torch::einsum("mij, mijkn->mijkn", {rf_inv_sqrt, tmp.h}) -
                    torch::einsum("mij, mijk, mijn->mijkn", {rf_inv_3_sqrt, tmp.d, tmp.d});

        // Return the new TensorMatHyperDual object
        return TensorMatHyperDual(r_sqrt, d_sqrt, h_sqrt);
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
            throw std::invalid_argument("Real part tensor must have dimensions [M, N, L]. Received: " + std::to_string(r.dim()));
        }
        if ( ddim == 0) {
            throw std::invalid_argument("Dual dimension must be greater than zero.");
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


    TensorMatHyperDual zeros_like(const torch::Tensor &x) const {
        // Validate the input tensor `x`
        if (!x.defined()) {
            throw std::invalid_argument("Input tensor `x` must be defined.");
        }

        // Validate the dual tensor `d`
        if (!d.defined() || d.dim() < 4) {
            throw std::runtime_error("Dual tensor `d` must be defined and have at least 4 dimensions.");
        }

        // Validate the hyperdual tensor `h`
        if (!h.defined() || h.dim() < 5) {
            throw std::runtime_error("Hyperdual tensor `h` must be defined and have at least 5 dimensions.");
        }

        // Validate batch size consistency
        if (x.size(0) != d.size(0) || x.size(0) != h.size(0)) {
            throw std::runtime_error("Batch size of `x` must match batch sizes of dual and hyperdual tensors.");
        }

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
        // Ensure all components are defined
        if (!this->r.defined() || !this->d.defined() || !this->h.defined()) {
          throw std::runtime_error("Cannot clone TensorMatHyperDual: one or more components are undefined.");
        }

        // Clone the real, dual, and hyperdual parts
        return TensorMatHyperDual(this->r.clone(), this->d.clone(), this->h.clone());
    }
    
    /**
     * Create an identity TensorMatHyperDual object with ones on the diagonal.
     * @param t1 The TensorMatHyperDual object.
     * @param t2 The TensorMatHyperDual object.
     * @param dim The dimension along which to concatenate the tensors.
     * @return A new TensorMatHyperDual object with ones on the diagonal and zeros elsewhere.
     */
    static TensorMatHyperDual cat(const TensorMatHyperDual& t1, const TensorMatHyperDual& t2, int dim = 2) {
        // Helper function to validate tensor shapes
        auto validate_shapes = [](const torch::Tensor& a, const torch::Tensor& b, int dim) {
            if (a.dim() != b.dim()) {
                return false;
            }
            for (int i = 0; i < a.dim(); ++i) {
                if (i != dim && a.size(i) != b.size(i)) {
                    return false;
                }
            }
            return true;
        };

        // Validate that the tensors can be concatenated
        if (!validate_shapes(t1.r, t2.r, dim) || !validate_shapes(t1.d, t2.d, dim) || !validate_shapes(t1.h, t2.h, dim)) {
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

        // Move t2 to the same device as t1 if necessary
        auto rt = t2.to(t1.r.device()).unsqueeze(0).expand({t1.r.size(0), t2.size(0), t2.size(1)});

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
        // Ensure tensors are defined
        if (!this->r.defined() || !this->d.defined() || !this->h.defined() ||
            !other.r.defined() || !other.d.defined() || !other.h.defined()) {
            throw std::invalid_argument("One or more tensors in TensorMatHyperDual objects are undefined.");
        }

        // Validate device compatibility
        if (this->r.device() != other.r.device() || this->d.device() != other.d.device() || this->h.device() != other.h.device()) {
            throw std::invalid_argument("TensorMatHyperDual objects must reside on the same device for addition.");
        }

        // Validate data type compatibility
        if (this->r.dtype() != other.r.dtype() || this->d.dtype() != other.d.dtype() || this->h.dtype() != other.h.dtype()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching data types for addition.");
        }

        // Validate dimension compatibility
        if (this->r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Real part dimensions do not match for addition.");
        }
        if (this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dual part dimensions do not match for addition.");
        }
        if (this->h.sizes() != other.h.sizes()) {
            throw std::invalid_argument("Hyperdual part dimensions do not match for addition.");
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
        // Ensure tensors are defined
        if (!this->r.defined() || !this->d.defined() || !this->h.defined() ||
            !other.r.defined() || !other.d.defined() || !other.h.defined()) {
            throw std::invalid_argument("One or more tensors in the objects are undefined.");
        }

        // Validate device compatibility
        if (this->r.device() != other.r.device() || this->d.device() != other.d.device() || this->h.device() != other.h.device()) {
            throw std::invalid_argument("TensorHyperDual and TensorMatHyperDual objects must reside on the same device.");
        }

        // Validate data type compatibility
        if (this->r.dtype() != other.r.dtype() || this->d.dtype() != other.d.dtype() || this->h.dtype() != other.h.dtype()) {
            throw std::invalid_argument("TensorHyperDual and TensorMatHyperDual objects must have matching data types.");
        }

        // Validate compatibility of dimensions
        if (this->r.size(0) != other.r.size(0) || this->r.size(2) != other.r.size(1)) {
            throw std::invalid_argument(
                "TensorHyperDual dimensions are incompatible with TensorMatHyperDual for addition.");
        }

        if (this->d.size(3) != other.d.size(2) || this->h.size(3) != other.h.size(2) || this->h.size(4) != other.h.size(3)) {
            throw std::invalid_argument(
                "TensorHyperDual dual and hyperdual dimensions are incompatible with TensorMatHyperDual for addition.");
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
        // Ensure the real part is defined and has a floating-point data type
        if (!this->r.defined()) {
            throw std::invalid_argument("Real part of TensorMatHyperDual is undefined.");
        }
        if (!this->r.is_floating_point()) {
            throw std::invalid_argument("Real part of TensorMatHyperDual must have a floating-point data type.");
        }

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
        // Ensure tensors are defined
        if (!this->r.defined() || !this->d.defined() || !this->h.defined() ||
            !other.r.defined() || !other.d.defined() || !other.h.defined()) {
            throw std::invalid_argument("One or more tensors in TensorMatHyperDual objects are undefined.");
        }

        // Validate device compatibility
        if (this->r.device() != other.r.device() || this->d.device() != other.d.device() || this->h.device() != other.h.device()) {
            throw std::invalid_argument("TensorMatHyperDual objects must reside on the same device for subtraction.");
        }

        // Validate data type compatibility
        if (this->r.dtype() != other.r.dtype() || this->d.dtype() != other.d.dtype() || this->h.dtype() != other.h.dtype()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching data types for subtraction.");
        }

        // Validate dimension compatibility
        if (this->r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Real part dimensions do not match for subtraction.");
        }
        if (this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dual part dimensions do not match for subtraction.");
        }
        if (this->h.sizes() != other.h.sizes()) {
            throw std::invalid_argument("Hyperdual part dimensions do not match for subtraction.");
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
        // Ensure the real part is defined and has a floating-point data type
        if (!this->r.defined()) {
            throw std::invalid_argument("Real part of TensorMatHyperDual is undefined.");
        }
        if (!this->r.is_floating_point()) {
            throw std::invalid_argument("Real part of TensorMatHyperDual must have a floating-point data type.");
        }

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

        // Validate dimension compatibility
        if (this->r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("TensorMatHyperDual objects must have matching dimensions for comparison.");
        }

        // Perform element-wise comparison of the real parts
        auto mask = this->r == other.r;

        // Squeeze unnecessary dimensions if applicable
        if (mask.dim() > 2 && mask.size(2) == 1) {
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
        auto dn = torch::einsum("mij, mij, mijk->mijk", {-r1, r2_inv2, d2}) +
                  torch::einsum("mij, mijk->mijk", {r2_inv, d1});

        auto d1d2 = torch::einsum("mijk, mijl->mijkl", {d1, d2});
        auto d2d2 = torch::einsum("mijl, mijk->mijkl", {d2, d2});
        
        
        // Hyperdual part of the result
        //-d1d2/r2^2 + 2r1/r2^3 * d2d2 + r1/r2^2 * h2 + r2^(-1) * h1 - r2^(-2) * d1d2
        auto hn = torch::einsum("mijkl, mij->mijkl", {-d1d2, r2_inv2}) +
          2 * torch::einsum("mijkl, mij->mijkl", {d2d2, r2_inv3}) +
          torch::einsum("mij, mij, mijkl->mijkl", {r1 ,r2_inv2, h2}) +
          torch::einsum("mij, mijkl->mijkl", {r2_inv, h1}) +
          torch::einsum("mijkl, mij->mijkl", {-d1d2, r2_inv2});
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
        auto h2 = other.h.unsqueeze(1);                  // Add singleton dimension

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
        // Ensure the scalar is not zero
        if (other == 0.0) {
            throw std::invalid_argument("Division by zero is not allowed in TensorMatHyperDual.");
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
        

        // Validate indices for the real part
        if (indices.size() > this->r.dim()) {
            throw std::invalid_argument("Too many indices for the real part of TensorMatHyperDual.");
        }

        // Index the real part
        auto r = this->r.index(indices);
        if (r.dim() == 2) {  // If a column is missing, unsqueeze dimension 2
            r = r.unsqueeze(2);
        }

        // Validate indices for the dual part
        if (indices.size() > this->d.dim()) {
            throw std::invalid_argument("Too many indices for the dual part of TensorMatHyperDual.");
        }

        // Index the dual part
        auto d = this->d.index(indices);
        if (d.dim() == 3) {  // If a column is missing, unsqueeze dimension 2
            d = d.unsqueeze(2);
        }

        // Validate indices for the hyperdual part
        if (indices.size() > this->h.dim()) {
            throw std::invalid_argument("Too many indices for the hyperdual part of TensorMatHyperDual.");
        }

        // Index the hyperdual part
        auto h = this->h.index(indices);
        if (h.dim() == 4) {  // If a column is missing, unsqueeze dimension 2
            h = h.unsqueeze(2);
        }

        // Validate indices: ensure all are slices or tensors
        for (const auto& idx : indices) {
            //Ensure the idx is not a integer
            if (idx.is_integer()) {
                throw std::invalid_argument("Indexing with integers is not allowed in TensorMatHyperDual.  Use Slice instead.");
            }
        }



        // Return a new TensorMatHyperDual object with the indexed components
        return TensorMatHyperDual(r, d, h);
    }

    /**
     * Index into the TensorMatHyperDual object along the first dimension using an integer index.
     *
     * This method extracts the slice corresponding to the specified index from the real, dual,
     * and hyperdual parts of the TensorMatHyperDual object along the batch dimension (first dimension),
     * ensuring that the output tensors preserve their dimensionality.
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
        auto real = this->r.index({index}).unsqueeze(0);      // Preserve batch dimension
        auto dual = this->d.index({index}).unsqueeze(0);      // Preserve batch dimension
        auto hyper = this->h.index({index}).unsqueeze(0);     // Preserve batch dimension

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

        // Validate that the mask is 1-dimensional
        if (mask.dim() != 1) {
            throw std::invalid_argument("The mask must be a 1-dimensional boolean tensor.");
        }

        // Validate that the mask matches the size of the batch dimension
        if (mask.size(0) != this->r.size(0)) {
            throw std::invalid_argument("The mask must have the same size as the batch dimension (" +
                                        std::to_string(this->r.size(0)) + ").");
        }

        // Validate that the mask is on the same device as the tensors
        if (mask.device() != this->r.device()) {
            throw std::invalid_argument("The mask must be on the same device as the TensorMatHyperDual components.");
        }

        // Perform indexing on the real, dual, and hyperdual parts
        auto real = r.index({mask});
        auto dual = d.index({mask});
        auto hyper = h.index({mask});

        // Return a new TensorMatHyperDual object with the indexed components
        return TensorMatHyperDual(real, dual, hyper);
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
        // Validate that tensors are defined
        if (!r.defined() || !d.defined() || !h.defined()) {
            throw std::runtime_error("One or more components of TensorMatHyperDual are undefined.");
        }

        // Validate that tensors are floating-point
        if (!r.is_floating_point() || !d.is_floating_point() || !h.is_floating_point()) {
            throw std::runtime_error("requires_grad_ can only be applied to floating-point tensors.");
        }

        // Set requires_grad for each tensor
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

        // Compute gradients for the real part
        if (r.requires_grad()) {
            if (grad_r) {
                if (grad_r->sizes() != r.sizes()) {
                    throw std::runtime_error("Shape mismatch: grad_r must have the same shape as r.");
                }
                if (grad_r->device() != r.device()) {
                    throw std::runtime_error("Device mismatch: grad_r must be on the same device as r.");
                }
                r.backward(grad_r.value());
            } else if (r.numel() == 1) {
                r.backward();
            } else {
                throw std::runtime_error("Gradient for 'r' must be specified when it has more than one element.");
            }
        }

        // Compute gradients for the dual part
        if (d.requires_grad()) {
            if (grad_d) {
                if (grad_d->sizes() != d.sizes()) {
                    throw std::runtime_error("Shape mismatch: grad_d must have the same shape as d.");
                }
                if (grad_d->device() != d.device()) {
                    throw std::runtime_error("Device mismatch: grad_d must be on the same device as d.");
                }
                d.backward(grad_d.value());
            } else if (d.numel() == 1) {
                d.backward();
            } else {
                throw std::runtime_error("Gradient for 'd' must be specified when it has more than one element.");
            }
        }

        // Compute gradients for the hyperdual part
        if (h.requires_grad()) {
            if (grad_h) {
                if (grad_h->sizes() != h.sizes()) {
                    throw std::runtime_error("Shape mismatch: grad_h must have the same shape as h.");
                }
                if (grad_h->device() != h.device()) {
                    throw std::runtime_error("Device mismatch: grad_h must be on the same device as h.");
                }
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
        // Validate einsum string
        auto pos = arg.find(",");
        auto pos2 = arg.find("->");
        if (pos == std::string::npos || pos2 == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: Must contain ',' and '->'.");
        }
        //Check for z and w in the einsum string
        if (arg.find("z") != std::string::npos || arg.find("w") != std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: 'z' and 'w' are reserved characters.");
        }

        // Validate tensors
        if (!first.r.defined() || !second.r.defined()) {
            throw std::invalid_argument("Input tensors must be defined.");
        }

        if (first.r.numel() == 0 || second.r.numel() == 0) {
            throw std::invalid_argument("Input tensors must be non-empty.");
        }

        // Compute real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Parse einsum arguments
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1);
        auto arg3 = arg.substr(pos2 + 2);

        // Compute dual part
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1 = torch::einsum(darg1, {first.d, second.r});
        auto darg2 = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto d2 = torch::einsum(darg2, {first.r, second.d});

        // Compute hyperdual part
        auto d1d2arg = arg1 + "z," + arg2 + "w->" + arg3 + "zw";
        auto d1d2 = torch::einsum(d1d2arg, {first.d, second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h, second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r, second.h});
        auto h = d1d2 + h1r2 + r1h2;

        // Return TensorHyperDual
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
        // Validate einsum string
        auto pos = arg.find(",");
        auto pos2 = arg.find("->");
        if (pos == std::string::npos || pos2 == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: Must contain ',' and '->'.");
        }
        //Check for z and w in the einsum string
        if (arg.find("z") != std::string::npos || arg.find("w") != std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: 'z' and 'w' are reserved characters.");
        }

        // Parse einsum arguments
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1);
        auto arg3 = arg.substr(pos2 + 2);

        if (first.r.sizes() != second.sizes()) {
            throw std::invalid_argument("Dimension mismatch between first.r and second for einsum.");
        }

        // Compute real part
        auto r = torch::einsum(arg, {first.r, second});

        // Compute dual part
        auto d1r1arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d = torch::einsum(d1r1arg, {first.d, second});

        // Compute hyperdual part
        auto h1r1arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h = torch::einsum(h1r1arg, {first.h, second});

        // Return TensorMatHyperDual
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
    static TensorHyperDual einsum(const std::string& arg, 
                                const TensorHyperDual& first, 
                                const TensorMatHyperDual& second) {
        // Validate einsum string
        auto pos = arg.find(",");
        auto pos2 = arg.find("->");
        if (pos == std::string::npos || pos2 == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: Must contain ',' and '->'.");
        }
        //Check for z and w in the einsum string
        if (arg.find("z") != std::string::npos || arg.find("w") != std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: 'z' and 'w' are reserved characters.");
        }

        // Parse einsum arguments
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1);
        auto arg3 = arg.substr(pos2 + 2);

        // Validate tensors
        if (!first.r.defined() || !second.r.defined()) {
            throw std::invalid_argument("Input tensors must be defined.");
        }

        // Compute real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Compute dual part
        auto d1r2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1r2 = torch::einsum(d1r2arg, {first.d, second.r});
        auto r1d2arg = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto r1d2 = torch::einsum(r1d2arg, {first.r, second.d});

        // Compute hyperdual part
        auto d1d2arg = arg1 + "z," + arg2 + "w->" + arg3 + "zw";
        auto d1d2 = torch::einsum(d1d2arg, {first.d, second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h, second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r, second.h});
        auto h = d1d2 + h1r2 + r1h2;

        // Return TensorHyperDual
        return TensorHyperDual(std::move(r), std::move(d1r2 + r1d2), std::move(h));
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
        // Validate einsum string
        if (arg.find(",") == std::string::npos || arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: Must contain ',' and '->'.");
        }
        if (arg.find("z") != std::string::npos || arg.find("w") != std::string::npos) {
            throw std::invalid_argument("Invalid einsum string: 'z' and 'w' are reserved characters.");
        }

        // Parse the einsum arguments
        auto pos = arg.find(",");
        auto pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 1, pos2 - pos - 1);
        auto arg3 = arg.substr(pos2 + 2);

        // Compute real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Compute dual part
        auto r1d2arg = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto r1d2 = torch::einsum(r1d2arg, {first.r, second.d});
        auto d1r2arg = arg1 + "z," + arg2 + "->" + arg3 + "z";
        auto d1r2 = torch::einsum(d1r2arg, {first.d, second.r});
        auto dual = r1d2 + d1r2;

        // Compute hyperdual part
        auto d1d2arg = arg1 + "z," + arg2 + "w->" + arg3 + "zw";
        auto d1d2 = torch::einsum(d1d2arg, {first.d, second.d});
        auto h1r2arg = arg1 + "zw," + arg2 + "->" + arg3 + "zw";
        auto h1r2 = torch::einsum(h1r2arg, {first.h, second.r});
        auto r1h2arg = arg1 + "," + arg2 + "zw->" + arg3 + "zw";
        auto r1h2 = torch::einsum(r1h2arg, {first.r, second.h});
        auto hyperdual = 2 * d1d2 + h1r2 + r1h2;

        // Return the resulting TensorMatHyperDual
        return TensorMatHyperDual(std::move(r), std::move(dual), std::move(hyperdual));
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
    void index_put_(const torch::Tensor& mask, const TensorMatHyperDual& value) {
        // Validate that the mask is a boolean tensor
        if (mask.scalar_type() != torch::kBool) {
            throw std::invalid_argument("The mask must be a boolean tensor.");
        }

        // Validate that the mask matches the batch size
        if (mask.size(0) != this->r.size(0)) {
            throw std::invalid_argument("The mask must have the same size as the batch dimension of the TensorMatHyperDual object.");
        }


        // Perform in-place updates on the real, dual, and hyperdual parts
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->h.index_put_({mask}, value.h);
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
    void index_put_(const torch::indexing::TensorIndex& mask, const TensorMatHyperDual& value) {

        // Perform in-place updates on the real, dual, and hyperdual parts
        // If there are index errors, they will be caught by the underlying index_put_ method
        this->r.index_put_({mask}, value.r.squeeze());
        this->d.index_put_({mask}, value.d.squeeze());
        this->h.index_put_({mask}, value.h.squeeze());
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
        // If there are index errors, they will be caught by the underlying index_put_ method

        // Perform in-place update on the real part
        this->r.index_put_({mask}, value);

        // Set the corresponding dual and hyperdual parts to zero
        this->d.index_put_({mask}, torch::zeros_like(this->d.index({mask})));
        this->h.index_put_({mask}, torch::zeros_like(this->h.index({mask})));

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


};




TensorMatHyperDual TensorHyperDual::eye() {


    // Create the real part as a batch of identity matrices
    auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});

    // Create the dual part as a zero tensor with the appropriate shape
    auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(1), this->d.size(2)}, this->d.options());

    // Create the hyper-dual part as a zero tensor with the appropriate shape
    auto h = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(1), this->d.size(2), this->h.size(3)}, this->h.options());

    // Return the resulting TensorMatHyperDual object
    return TensorMatHyperDual(r, d, h);
}


/**
 * @brief Adds a singleton dimension to the TensorDual and returns it as a TensorMatDual.
 *
 * This method adds a singleton dimension at the specified position to both the real (`r`) and
 * dual (`d`) tensors of the `TensorDual` object. The resulting tensors are encapsulated in
 * a `TensorMatDual` object.
 *
 * @param dim The dimension along which to add the singleton dimension.
 * @return TensorMatDual A new TensorMatDual object with the added singleton dimension.
 * @throws std::invalid_argument If the real or dual tensors are undefined or the specified dimension is invalid.
 */
TensorMatDual TensorDual::unsqueeze(int dim) {

    // Normalize negative dimension
    if (dim < 0) {
        dim += this->r.dim() + 1; // Normalize negative dimension
    }

    // Validate that the dimension is within the valid range
    if (dim < 0 || dim > this->r.dim()) {
        throw std::invalid_argument("Cannot unsqueeze: dimension is out of bounds.");
    }

    // Add a singleton dimension to both tensors
    auto r = this->r.unsqueeze(dim);
    auto d = this->d.unsqueeze(dim);

    // Return the resulting TensorMatDual
    return TensorMatDual(std::move(r), std::move(d));
}


/**
 * @brief Creates a batch identity matrix for the real part and initializes the dual part as zeros.
 *
 * This method generates a batch identity matrix for the real (`r`) tensor and initializes the dual
 * (`d`) tensor as zeros with an appropriate shape. The shapes are determined based on the current
 * dimensions of the `r` and `d` tensors.
 *
 * @return TensorMatDual A new TensorMatDual object with the real part as a batch identity matrix and
 * the dual part as zeros.
 * @throws std::runtime_error If the real or dual tensors are undefined or have invalid dimensions.
 */
TensorMatDual TensorDual::eye() {


    // Create the real part as a batch of identity matrices
    auto r = torch::eye(this->r.size(1), this->r.options()).repeat({this->r.size(0), 1, 1});

    // Create the dual part as a zero tensor with the appropriate shape
    auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(1), this->d.size(2)}, this->d.options());

    // Return the resulting TensorMatDual
    return TensorMatDual(r, d);
}


/**
 * @brief Overloads the `*` operator for the multiplication of a torch::Tensor and a TensorDual.
 *
 * This function computes the element-wise product of a `torch::Tensor` and the real (`r`) part of
 * a `TensorDual` object. It also scales the dual (`d`) part of the `TensorDual` object by unsqueezing
 * the `torch::Tensor` for proper broadcasting.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined or incompatible for broadcasting.
 */
TensorDual operator*(const torch::Tensor& tensor, const TensorDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot multiply: the torch::Tensor is undefined.");
    }


    // Compute the real and dual parts
    auto real = tensor * td.r;
    auto dual = tensor.unsqueeze(-1) * td.d;

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}


/**
 * @brief Overloads the `*` operator for the multiplication of a torch::Tensor and a TensorHyperDual.
 *
 * This function computes the element-wise product of a `torch::Tensor` and the real (`r`) part of
 * a `TensorHyperDual` object. It also scales the dual (`d`) and hyper-dual (`h`) parts of the
 * `TensorHyperDual` object by unsqueezing the `torch::Tensor` for proper broadcasting.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorHyperDual object acting as the right-hand operand.
 * @return TensorHyperDual A new TensorHyperDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined or incompatible for broadcasting.
 */
TensorHyperDual operator*(const torch::Tensor& tensor, const TensorHyperDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot multiply: the torch::Tensor is undefined.");
    }


    // Compute the real, dual, and hyper-dual parts
    auto real = tensor * td.r;
    auto dual = tensor.unsqueeze(-1) * td.d;
    auto hyper = tensor.unsqueeze(-1).unsqueeze(-1) * td.h;

    // Return the resulting TensorHyperDual
    return TensorHyperDual(std::move(real), std::move(dual), std::move(hyper));
}


/**
 * @brief Overloads the `/` operator for dividing a torch::Tensor by a TensorDual.
 *
 * This function computes the element-wise division of a `torch::Tensor` by the real (`r`) part of
 * a `TensorDual` object. It also computes the corresponding dual part using the chain rule.
 *
 * @param tensor The torch::Tensor acting as the numerator.
 * @param td The TensorDual object acting as the denominator.
 * @return TensorDual A new TensorDual object representing the result of the division.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined or incompatible for broadcasting.
 */
TensorDual operator/(const torch::Tensor& tensor, const TensorDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot divide: the torch::Tensor is undefined.");
    }

    // Perform division for the real part
    auto r = tensor / td.r;

    // Compute the dual part using the chain rule
    auto d = -(tensor / td.r.square()).unsqueeze(-1) * td.d;

    // Return the resulting TensorDual
    return TensorDual(std::move(r), std::move(d));
}



/**
 * @brief Overloads the `/` operator for dividing a torch::Tensor by a TensorHyperDual.
 *
 * This function computes the element-wise division of a `torch::Tensor` by the real (`r`) part of
 * a `TensorHyperDual` object. It also computes the corresponding dual (`d`) and hyper-dual (`h`) parts
 * using the chain rule and tensor operations.
 *
 * @param tensor The torch::Tensor acting as the numerator.
 * @param td The TensorHyperDual object acting as the denominator.
 * @return TensorHyperDual A new TensorHyperDual object representing the result of the division.
 * @throws std::invalid_argument If the input tensors are undefined or incompatible for broadcasting.
 */
TensorHyperDual operator/(const torch::Tensor& tensor, const TensorHyperDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot divide: the torch::Tensor is undefined.");
    }
    if (!td.r.defined() || !td.d.defined() || !td.h.defined()) {
        throw std::invalid_argument("Cannot divide: the TensorHyperDual tensors are undefined.");
    }

    // Alias the components of TensorHyperDual for clarity
    auto r1 = tensor;
    auto r2 = td.r;  // Real part of the denominator
    auto d2 = td.d;  // Dual part of the denominator
    auto h2 = td.h;  // Hyper-dual part of the denominator

    // Compute the real part
    auto r = r1 / r2;

    // Compute the dual part using the chain rule
    auto d = -(r1 / r2.square()).unsqueeze(-1) * d2;

    // Compute the hyper-dual part using the chain rule
    auto h = torch::einsum("mi, mi, mij, mik -> mijk", {r1, r2.pow(-3), d2, d2}) -
             torch::einsum("mi, mi, mijk -> mijk", {r1, r2.pow(-2), h2});

    // Return the resulting TensorHyperDual
    return TensorHyperDual(std::move(r), std::move(d), std::move(h));
}

/**
 * @brief Overloads the `+` operator for adding a torch::Tensor to a TensorDual.
 *
 * This function computes the element-wise addition of a `torch::Tensor` to the real (`r`) part of
 * a `TensorDual` object. The dual (`d`) part of the `TensorDual` remains unchanged.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the addition.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined.
 */
TensorDual operator+(const torch::Tensor& tensor, const TensorDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot add: the torch::Tensor is undefined.");
    }

    // Compute the real part by adding the tensor to the real part of TensorDual
    auto real = tensor + td.r;

    // The dual part remains unchanged; clone it to ensure the result is independent
    auto dual = td.d.clone();

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}


/**
 * @brief Overloads the `+` operator for adding a torch::Tensor to a TensorHyperDual.
 *
 * This function computes the element-wise addition of a `torch::Tensor` to the real (`r`) part of
 * a `TensorHyperDual` object. The dual (`d`) and hyper-dual (`h`) parts of the `TensorHyperDual`
 * remain unchanged and are cloned to ensure independence.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorHyperDual object acting as the right-hand operand.
 * @return TensorHyperDual A new TensorHyperDual object representing the result of the addition.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined.
 */
TensorHyperDual operator+(const torch::Tensor& tensor, const TensorHyperDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot add: the torch::Tensor is undefined.");
    }

    // Compute the real part by adding the tensor to the real part of TensorHyperDual
    auto real = tensor + td.r;

    // Clone the dual and hyper-dual parts to ensure the result is independent
    auto dual = td.d.clone();
    auto hyper_dual = td.h.clone();

    // Return the resulting TensorHyperDual
    return TensorHyperDual(std::move(real), std::move(dual), std::move(hyper_dual));
}

/**
 * @brief Overloads the `+` operator for adding a scalar to a TensorDual.
 *
 * This function computes the element-wise addition of a scalar to the real (`r`) part of a
 * `TensorDual` object. The dual (`d`) part remains unchanged and is cloned to ensure independence.
 *
 * @param scalar The scalar (double) acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the addition.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined.
 */
TensorDual operator+(const double& scalar, const TensorDual& td) {

    // Compute the real part by adding the scalar to the real part of TensorDual
    auto real = td.r + scalar;

    // Clone the dual part to ensure the result is independent of the input TensorDual
    auto dual = td.d.clone();

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}



/**
 * @brief Overloads the `+` operator for adding a scalar to a TensorHyperDual.
 *
 * This function computes the element-wise addition of a scalar to the real (`r`) part of a
 * `TensorHyperDual` object. The dual (`d`) and hyper-dual (`h`) parts remain unchanged and
 * are cloned to ensure independence.
 *
 * @param scalar The scalar (double) acting as the left-hand operand.
 * @param td The TensorHyperDual object acting as the right-hand operand.
 * @return TensorHyperDual A new TensorHyperDual object representing the result of the addition.
 * @throws std::invalid_argument If the TensorHyperDual's real, dual, or hyper-dual tensors are undefined.
 */
TensorHyperDual operator+(const double& scalar, const TensorHyperDual& td) {

    // Compute the real part by adding the scalar to the real part of TensorHyperDual
    auto real = td.r + scalar;

    // Clone the dual and hyper-dual parts to ensure the result is independent
    auto dual = td.d.clone();
    auto hyper_dual = td.h.clone();

    // Return the resulting TensorHyperDual
    return TensorHyperDual(std::move(real), std::move(dual), std::move(hyper_dual));
}


/**
 * @brief Overloads the `-` operator for subtracting a TensorDual from a torch::Tensor.
 *
 * This function computes the element-wise subtraction of the real (`r`) part of a `TensorDual`
 * object from a `torch::Tensor`. The dual (`d`) part of the `TensorDual` is negated and returned
 * as part of the result.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the subtraction.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined.
 */
TensorDual operator-(const torch::Tensor& tensor, const TensorDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot subtract: the torch::Tensor is undefined.");
    }
    // Compute the real part by subtracting the real part of TensorDual from the tensor
    auto real = tensor - td.r;

    // Negate the dual part and clone it to ensure the result is independent
    auto dual = -td.d.clone();

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}




/**
 * @brief Overloads the `-` operator for subtracting a TensorHyperDual from a torch::Tensor.
 *
 * This function computes the element-wise subtraction of the real (`r`) part of a `TensorHyperDual`
 * object from a `torch::Tensor`. The dual (`d`) and hyper-dual (`h`) parts of the `TensorHyperDual`
 * are negated and returned as part of the result.
 *
 * @param tensor The torch::Tensor acting as the left-hand operand.
 * @param td The TensorHyperDual object acting as the right-hand operand.
 * @return TensorHyperDual A new TensorHyperDual object representing the result of the subtraction.
 * @throws std::invalid_argument If the `tensor` or `td` tensors are undefined.
 */
TensorHyperDual operator-(const torch::Tensor& tensor, const TensorHyperDual& td) {
    // Validate that the tensors are defined
    if (!tensor.defined()) {
        throw std::invalid_argument("Cannot subtract: the torch::Tensor is undefined.");
    }

    // Compute the real part by subtracting the real part of TensorHyperDual from the tensor
    auto real = tensor - td.r;

    // Negate the dual and hyper-dual parts, cloning them to ensure the result is independent
    auto dual = -td.d.clone();
    auto hyper_dual = -td.h.clone();

    // Return the resulting TensorHyperDual
    return TensorHyperDual(std::move(real), std::move(dual), std::move(hyper_dual));
}


/**
 * @brief Overloads the `*` operator for multiplying a TensorDual by a TensorMatDual.
 *
 * This function computes the product of a `TensorDual` object and a `TensorMatDual` object.
 * Depending on the dimensions of the input tensors, it handles left or right multiplication
 * and computes the real (`r`) and dual (`d`) parts using Einstein summation notation.
 *
 * @param td The TensorDual object acting as the left-hand operand.
 * @param other The TensorMatDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the dimensions of `td` or `other` are incompatible for multiplication.
 */
TensorDual operator*(const TensorDual& td, const TensorMatDual& other) {

    torch::Tensor r, d;

    // Determine the multiplication type based on dimensions
    if (td.r.size(1) == other.r.size(1)) {
        // Left multiplication
        r = torch::einsum("mi, mij -> mj", {td.r, other.r});
        d = torch::einsum("mi, mijn -> mjn", {td.r, other.d}) +
            torch::einsum("min, mij -> mjn", {td.d, other.r});
    } else if (td.r.size(1) == other.r.size(2)) {
        // Right multiplication
        r = torch::einsum("mi, mji -> mj", {td.r, other.r});
        d = torch::einsum("mi, mjin -> mjn", {td.r, other.d}) +
            torch::einsum("min, mji -> mjn", {td.d, other.r});
    } else {
        throw std::invalid_argument("Cannot multiply: incompatible dimensions between TensorDual and TensorMatDual.");
    }

    // Return the resulting TensorDual
    return TensorDual(std::move(r), std::move(d));
}

/**
 * @brief Overloads the `*` operator for multiplying a TensorMatDual by a TensorDual.
 *
 * This function computes the product of a `TensorMatDual` object and a `TensorDual` object.
 * Depending on the dimensions of the input tensors, it determines the appropriate multiplication
 * operation and computes the real (`r`) and dual (`d`) parts using Einstein summation notation.
 *
 * @param tmd The TensorMatDual object acting as the left-hand operand.
 * @param other The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the dimensions of `tmd` or `other` are incompatible for multiplication.
 */
TensorDual operator*(const TensorMatDual& tmd, const TensorDual& other) {

    torch::Tensor r, d;

    // Determine the multiplication type based on dimensions
    if (tmd.r.size(2) == other.r.size(1)) {
        // Compatible for matrix multiplication
        r = torch::einsum("mij, mj -> mi", {tmd.r, other.r});
        d = torch::einsum("mijn, mj -> min", {tmd.d, other.r}) +
            torch::einsum("mij, mjn -> min", {tmd.r, other.d});
    } else if (tmd.r.size(1) == other.r.size(1)) {
        // Compatible for transposed multiplication
        r = torch::einsum("mij, mi -> mj", {tmd.r, other.r});
        d = torch::einsum("mijn, mi -> mjn", {tmd.d, other.r}) +
            torch::einsum("mij, min -> mjn", {tmd.r, other.d});
    } else {
        throw std::invalid_argument("Cannot multiply: incompatible dimensions between TensorMatDual and TensorDual.");
    }

    // Return the resulting TensorDual
    return TensorDual(std::move(r), std::move(d));
}

/**
 * @brief Overloads the `*` operator for element-wise multiplication of two TensorMatDual objects.
 *
 * This function computes the element-wise product of the real (`r`) and dual (`d`) parts of two
 * `TensorMatDual` objects. The dual part is computed using the product rule.
 *
 * @param lhs The left-hand operand TensorMatDual.
 * @param rhs The right-hand operand TensorMatDual.
 * @return TensorMatDual A new TensorMatDual object representing the result of the element-wise multiplication.
 * @throws std::invalid_argument If the dimensions of `lhs` and `rhs` are incompatible for element-wise multiplication.
 */
TensorMatDual operator*(const TensorMatDual& lhs, const TensorMatDual& rhs) {

    // Check for dimension compatibility
    if (lhs.r.sizes() != rhs.r.sizes()) {
        throw std::invalid_argument("Cannot multiply: incompatible dimensions for TensorMatDual objects.");
    }

    // Compute the real and dual parts
    auto r = lhs.r * rhs.r; // Element-wise multiplication of the real parts
    auto d = torch::einsum("mij, mijn -> mijn", {lhs.r, rhs.d}) +
             torch::einsum("mijn, mij -> mijn", {lhs.d, rhs.r});

    // Return the resulting TensorMatDual
    return TensorMatDual(std::move(r), std::move(d));
}



/**
 * @brief Overloads the `-` operator for subtracting a TensorDual from a scalar.
 *
 * This function computes the element-wise subtraction of the real (`r`) part of a `TensorDual`
 * object from a scalar. The dual (`d`) part is negated and returned as part of the result.
 *
 * @param scalar The scalar (int) acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the subtraction.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined.
 */
TensorDual operator-(const int& scalar, const TensorDual& td) {

    // Create a tensor from the scalar with the same options as TensorDual's real tensor
    auto scalar_tensor = torch::tensor({scalar}, td.r.options());

    // Compute the real part by subtracting the real part of TensorDual from the scalar
    auto real = scalar_tensor - td.r;

    // Negate the dual part and ensure the result is independent
    auto dual = -td.d;

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}

/**
 * @brief Overloads the `/` operator for dividing a scalar by a TensorDual.
 *
 * This function computes the element-wise division of a scalar by the real (`r`) part of a `TensorDual` object.
 * The dual (`d`) part is computed using the chain rule.
 *
 * @param scalar The scalar (double) acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the division.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined.
 */
TensorDual operator/(double& scalar, const TensorDual& td) {

    // Compute the real part by dividing the scalar by the real part of TensorDual
    auto r = scalar / td.r;

    // Compute the dual part using the chain rule, and apply unsqueeze to match dimensions
    auto d = -(scalar / (td.r.square())).unsqueeze(-1) * td.d;

    // Return the resulting TensorDual
    return TensorDual(r, d);
}


/**
 * @brief Overloads the `*` operator for multiplying a scalar by a TensorDual.
 *
 * This function computes the element-wise multiplication of a scalar with the real (`r`) and dual (`d`) 
 * parts of a `TensorDual` object. The dual part is also scaled by the scalar.
 *
 * @param scalar The scalar (double) acting as the left-hand operand.
 * @param td The TensorDual object acting as the right-hand operand.
 * @return TensorDual A new TensorDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined.
 */
TensorDual operator*(const double& scalar, const TensorDual& td) {

    // Compute the real part by multiplying the real part of TensorDual by the scalar
    auto real = td.r * scalar;

    // Compute the dual part by multiplying the dual part of TensorDual by the scalar
    auto dual = td.d * scalar;

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}

/**
 * @brief Overloads the `*` operator for multiplying a scalar by a TensorMatDual.
 *
 * This function computes the element-wise multiplication of a scalar with the real (`r`) and dual (`d`) 
 * parts of a `TensorMatDual` object. The dual part is also scaled by the scalar.
 *
 * @param scalar The scalar (double) acting as the left-hand operand.
 * @param td The TensorMatDual object acting as the right-hand operand.
 * @return TensorMatDual A new TensorMatDual object representing the result of the multiplication.
 * @throws std::invalid_argument If the TensorMatDual's real or dual tensors are undefined.
 */
TensorMatDual operator*(const double& scalar, const TensorMatDual& td) {

    // Compute the real part by multiplying the real part of TensorMatDual by the scalar
    auto real = td.r * scalar;

    // Compute the dual part by multiplying the dual part of TensorMatDual by the scalar
    auto dual = td.d * scalar;

    // Return the resulting TensorMatDual
    return TensorMatDual(std::move(real), std::move(dual));
}


/**
 * @brief Computes the power of a TensorDual raised to another TensorDual.
 *
 * Given `x = a + bε` and `y = c + dε`, the formula for `x^y` is:
 * 
 *    x^y = a^c + (a^(c-1) * b * c + a^c * d * log(a))ε
 * 
 * This function computes both the real and dual parts of the result using the above formula.
 *
 * @param base The base TensorDual (a + bε).
 * @param exponent The exponent TensorDual (c + dε).
 * @return TensorDual A new TensorDual object representing the result of the exponentiation.
 * @throws std::invalid_argument If the real or dual tensors of `base` or `exponent` are undefined.
 */
TensorDual pow(const TensorDual& base, const TensorDual& exponent) {

    // Extract real and dual parts of base and exponent
    auto a = base.r;
    auto b = base.d;
    auto c = exponent.r;
    auto d = exponent.d;

    // Compute the real part using exponentiation
    auto real = torch::pow(a, c);

    // Compute the dual part using the chain rule
    auto dual = torch::einsum("mi, mij->mij", {real * torch::log(a), d}) + 
                torch::einsum("mi, mij->mij", {real * (c / a), b});

    // Return the resulting TensorDual
    return TensorDual(std::move(real), std::move(dual));
}


/**
 * @brief Computes the element-wise maximum of two TensorDual objects.
 *
 * This function computes the element-wise maximum of the real (`r`) parts of two `TensorDual` objects.
 * The dual (`d`) parts are also selected based on which `r` part was greater at each position.
 *
 * @param lhs The left-hand TensorDual operand.
 * @param rhs The right-hand TensorDual operand.
 * @return TensorDual A new TensorDual object representing the result of the element-wise maximum.
 * @throws std::invalid_argument If the real or dual tensors of the input objects are undefined.
 */
TensorDual max(const TensorDual& lhs, const TensorDual& rhs) {

    // Compute the real part (element-wise maximum)
    auto r = torch::max(lhs.r, rhs.r);

    // Create a tensor for the dual part initialized with zeros (same size as lhs.d)
    auto d = torch::zeros_like(lhs.d);

    // Create a mask where lhs.r < rhs.r
    auto maskrgt = lhs.r < rhs.r;

    // Set the dual part based on the mask: if maskrgt is true, use rhs.d; otherwise, use lhs.d
    d.index_put_({maskrgt}, rhs.d.index({maskrgt}));
    d.index_put_({~maskrgt}, lhs.d.index({~maskrgt}));

    // Return the resulting TensorDual
    return TensorDual(r, d);
}

/**
 * @brief Computes the element-wise maximum of a TensorDual and a torch::Tensor.
 *
 * This function computes the element-wise maximum of the real (`r`) part of a `TensorDual` 
 * object and a `torch::Tensor`. The dual (`d`) part of the `TensorDual` is selected based on 
 * which tensor had the maximum value at each position.
 *
 * @param lhs The left-hand TensorDual object.
 * @param rhs The right-hand tensor (torch::Tensor).
 * @return TensorDual A new TensorDual object representing the result of the element-wise maximum.
 * @throws std::invalid_argument If the dimensions of `lhs.r` and `rhs` are incompatible.
 */
TensorDual max(const TensorDual& lhs, const torch::Tensor& rhs) {

    // Create a mask where the real part of lhs is greater than rhs
    auto mask = lhs.r > rhs;

    // Initialize tensors for the result of real and dual parts
    auto resr = torch::zeros_like(lhs.r);
    auto resd = torch::zeros_like(lhs.d);    

    // Assign the maximum real part values based on the mask
    resr.index_put_({mask}, lhs.r.index({mask}));
    resr.index_put_({~mask}, rhs.index({~mask}));

    // Assign the corresponding dual part based on the mask
    resd.index_put_({mask}, lhs.d.index({mask}));
    resd.index_put_({~mask}, torch::zeros_like(lhs.d).index({~mask}));

    // Return the resulting TensorDual
    return TensorDual(resr, resd);
}



/**
 * @brief Computes the element-wise minimum of two TensorDual objects.
 *
 * This function computes the element-wise minimum of the real (`r`) parts of two `TensorDual` objects.
 * The dual (`d`) part is selected based on which `r` part was smaller at each position.
 *
 * @param lhs The left-hand TensorDual operand.
 * @param rhs The right-hand TensorDual operand.
 * @return TensorDual A new TensorDual object representing the result of the element-wise minimum.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined or if dimensions are incompatible.
 */
TensorDual min(const TensorDual& lhs, const TensorDual& rhs) {

    // Compute the real part (element-wise minimum)
    auto r = torch::min(lhs.r, rhs.r);

    // Create a mask where lhs.r is smaller than rhs.r
    auto maskl = lhs.r < rhs.r;

    // Initialize the dual part tensor (same size as lhs.d)
    auto d = torch::zeros_like(lhs.d);

    // Set the dual part based on the mask: if lhs.r is smaller, use lhs.d; otherwise, use rhs.d
    d.index_put_({maskl}, lhs.d.index({maskl}));
    d.index_put_({~maskl}, rhs.d.index({~maskl}));

    // Return the resulting TensorDual
    return TensorDual(r, d);
}




/**
 * @brief Computes the element-wise minimum of a TensorDual and a torch::Tensor.
 *
 * This function computes the element-wise minimum of the real (`r`) part of a `TensorDual`
 * and a `torch::Tensor`. The dual (`d`) part of the `TensorDual` is selected based on which `r` part
 * was smaller at each position.
 *
 * @param lhs The left-hand TensorDual operand.
 * @param rhs The right-hand tensor (torch::Tensor).
 * @return TensorDual A new TensorDual object representing the result of the element-wise minimum.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined or if dimensions are incompatible.
 */
TensorDual min(const TensorDual& lhs, const torch::Tensor& rhs) {

    // Create a mask where the real part of lhs is smaller than rhs
    auto mask = lhs.r < rhs;

    // Create tensors for the result, initialized to zeros
    auto resr = torch::zeros_like(lhs.r);
    auto resd = torch::zeros_like(lhs.d);

    // Set the real part based on the mask
    resr.index_put_({mask}, lhs.r.index({mask}));
    resr.index_put_({~mask}, rhs.index({~mask}));

    // Set the dual part based on the mask: if lhs.r is smaller, use lhs.d; otherwise, use rhs.d
    resd.index_put_({mask}, lhs.d.index({mask}));
    resd.index_put_({~mask}, torch::zeros_like(lhs.d).index({~mask}));

    // Return the resulting TensorDual
    return TensorDual(resr, resd);
}



/**
 * @brief Computes the sign of a TensorDual object.
 *
 * This function computes the element-wise sign of the real (`r`) part of a `TensorDual` object. The 
 * dual (`d`) part is set to zero except when the real part is zero, in which case it matches the real 
 * part's sign.
 *
 * @param td The TensorDual object.
 * @return TensorDual A new TensorDual object representing the sign of the real and dual parts.
 * @throws std::invalid_argument If the TensorDual's real or dual tensors are undefined.
 */
static TensorDual sign(TensorDual& td) {

    // Compute the real part: sign of the real part of TensorDual
    auto r = torch::sign(td.r);

    // Create a mask where the real part is zero
    auto maskz = td.r == 0;

    // Create the dual part tensor initialized to zero
    auto d = torch::zeros_like(td.d);

    // If there are any elements where the real part is zero, set the corresponding dual part to the real part
    if (maskz.any().item<bool>()) {
        d.index_put_({maskz}, r.index({maskz}));  // Dual part is the same as the real part when real part is zero
    }

    // Return the resulting TensorDual
    return TensorDual(r, d);
}


/**
 * @brief Computes the pow of a TensorDual object with a torch::Tensor.
 * @param base The base TensorDual object.
 * @param exponent The exponent torch::Tensor.
 * @return TensorDual A new TensorDual object representing the result of the exponentiation.
 */
TensorDual pow(const TensorDual& base, const torch::Tensor& exponent)
{

    // Compute the real part using torch::pow
    auto r = torch::pow(base.r, exponent);
    torch::Tensor d=torch::einsum("mij, mi->mij", {base.d, exponent * base.r.pow(exponent - 1)});
    // Compute the dual part using the chain rule

    // Return the resulting TensorDual
    return TensorDual(r, d);
}

/**
 * @brief Computes the power of a TensorDual object raised to a scalar.
 * @param base The base TensorDual object.
 * @param exponent The exponent (double).
 * @return TensorDual A new TensorDual object representing the result of the exponentiation.
 * @throws std::invalid_argument If the base TensorDual's tensors are undefined.
 */
TensorDual pow(const TensorDual& base, double exponent) 
{
    //reshape the scalar to match the dimensions of the base tensor
    auto exp = torch::full_like(base.r, exponent);
    // Compute the real part using torch::pow
    auto r = torch::pow(base.r, exp);

    // Compute the dual part using the chain rule
    auto d = base.d*exp.unsqueeze(-1) *torch::pow(base.r, exp - 1).unsqueeze(-1);

    // Return the resulting TensorDual
    return TensorDual(r, d);
}



/**
 * @brief Computes the generalized outer product (ger) of two TensorDual objects.
 *
 * This function computes the outer product of two `TensorDual` objects using Einstein summation notation.
 * It calculates both the real (`r`) and dual (`d`) parts using the following formulas:
 * 
 *   r = x.r @ y.r (outer product of real parts)
 *   d1 = x.r @ y.d (outer product of real part of x and dual part of y)
 *   d2 = x.d @ y.r (outer product of dual part of x and real part of y)
 * 
 * The dual part is computed as d = d1 + d2.
 *
 * @param x The first TensorDual object.
 * @param y The second TensorDual object.
 * @return TensorMatDual A new TensorMatDual object representing the result of the outer product.
 * @throws std::invalid_argument If the dimensions of `x.r`, `y.r`, `x.d`, or `y.d` are incompatible for the operation.
 */
TensorMatDual ger(const TensorDual& x, const TensorDual& y) {

    // Compute the real part using einsum: x.r @ y.r
    auto r = torch::einsum("mj, mi->mij", {x.r, y.r});

    // Compute the dual part: two terms from outer products
    auto d1 = torch::einsum("mj, mik->mijk", {x.r, y.d});
    auto d2 = torch::einsum("mjk, mi->mijk", {x.d, y.r});

    // Return the resulting TensorMatDual with real part r and dual part d1 + d2
    return TensorMatDual(r, d1 + d2);
}


}



#endif