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


class TensorDual {

public:
    // Members
    torch::Tensor r;
    torch::Tensor d;
    //Add a dtype
    torch::Dtype dtype;
    torch::Device device_= torch::kCPU;


public:
    // Constructor by reference
    TensorDual(torch::Tensor& r, torch::Tensor& d) noexcept {
        assert(r.dim() == 2 && d.dim() == 3 && "r must be 2D and d must be 3D");
        this->r = r;
        this->d = d;
        this->dtype = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }

    
    //Constructor using move semantics
    TensorDual(torch::Tensor&& rin, torch::Tensor&& din) noexcept : r(std::move(rin)), d(std::move(din)) {
        assert(r.dim() == 2 && d.dim() == 3 && "r must be 2D and d must be 3D");    
    }


    TensorDual() {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

        // Create zero tensors with the specified options
        torch::Tensor rl{torch::zeros({1, 1}, options)};
        torch::Tensor dl{torch::zeros({1, 1, 1}, options)};
        TensorDual(rl, dl);

    }

    friend std::ostream& operator<<(std::ostream& os, const TensorDual& obj){
        os << "r: " << obj.r << std::endl;
        os << "d: " << obj.d << std::endl;
        return os;
    }

    torch::Device device() const {
        return device_;
    }

    TensorDual to(const torch::Device& device) {
        auto rn = r.to(device);
        auto rd = d.to(device);
        return TensorDual(rn, rd);
    }
    
    void set_requires_grad(bool req_grad) {
        r.set_requires_grad(req_grad);
        d.set_requires_grad(req_grad);
    }

    void requires_grad_(bool req_grad) {
        r.requires_grad_(req_grad);
        d.requires_grad_(req_grad);
    }

    void backward() {
        r.backward();
        d.backward();
    }


    /**
     * Calculates the gradient of the TensorDual object
     * with respect to the input Dual tensor x
     */
    void backward(TensorDual& grad) {
        r.backward(grad.r);
        d.backward(grad.d);
    }
    //Gradient will expand the dimensionality of the vector tensor to matrix tensor
    TensorDual grad() {
        return TensorDual(r.grad().clone(), d.grad().clone());
    }



    // Static member function to replicate the @classmethod 'create'
    static TensorDual create(torch::Tensor& r) {
        auto l = r.size(1); // number of leaves N
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(l); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);

        // Set the last dimension as the identity matrix
        for (int64_t i = 0; i < l; ++i) {
            ds.index({torch::indexing::Slice(), i, i}) = 1.0;
        }

        return TensorDual(r, ds);
    }

        // Static member function to create a TensorDual with zeros
    static TensorDual zeros_like(const TensorDual& x) {
        auto r = torch::zeros_like(x.r);
        auto ds = torch::zeros_like(x.d);
        return TensorDual(r, ds);
    }


    /*
     * Create a dual tensor based on the tensor x as real part
     * and the current tensor's dual part adjusted to match the shape of x
     */
    TensorDual zeros_like(const torch::Tensor& x) {
        auto r = torch::zeros_like(x, x.dtype());
        int M  = r.size(0);
        int nr = r.size(1);
        int nd = d.size(2);
        torch::Tensor ds;
        if (r.dtype() == torch::kBool) {
            ds = torch::zeros({M, nr, nd}, torch::kFloat64);
        } else {
            ds = torch::zeros({M, nr, nd}, x.dtype());
        }
        return TensorDual(r, ds);
    }

    TensorDual zeros_like()
    {
        auto r = torch::zeros_like(this->r);
        auto d = torch::zeros_like(this->d);
        return TensorDual(r, d);
    }

    // Static member function to create a TensorDual with ones
    static TensorDual ones_like(const TensorDual& x) {
        auto r = torch::ones_like(x.r, x.r.dtype());
        torch::Tensor ds;
        if (r.dtype() == torch::kBool) {
            ds = torch::zeros_like(x.d, torch::kFloat64);
        } else {
            ds = torch::zeros_like(x.d, x.r.dtype());
        }
        return TensorDual(r, ds);
    }

     // Static member function to create a boolean tensor like the real part of x
    static torch::Tensor bool_like(const TensorDual& x) {
        return torch::zeros_like(x.r, torch::dtype(torch::kBool).device(x.r.device()));
    }

    static TensorDual createZero(torch::Tensor& r, int ddim) {
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(ddim); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        return TensorDual(r, ds);
    }

    static TensorDual createZero(torch::Tensor&& r, int ddim) {
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(ddim); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        return TensorDual(r, ds);
    }

    static TensorDual empty_like(const TensorDual& x) {
        auto r = torch::empty_like(x.r);
        auto ds = torch::empty_like(x.d);
        return TensorDual(r, ds);
    }


        // Static member function to concatenate TensorDual objects
    static TensorDual cat(const std::vector<TensorDual>& args) {
        // Check to make sure all elements of args are TensorDual
        std::vector<torch::Tensor> r_tensors;
        std::vector<torch::Tensor> d_tensors;

        for (const auto& a : args) {
            r_tensors.push_back(a.r);
            d_tensors.push_back(a.d);
        }

        auto r = torch::cat(r_tensors, 1);
        auto d = torch::cat(d_tensors, 1);

        return TensorDual(r, d);
    }


    /**
     * Static member function to perform einsum on two TensorDual objects
     * Note this is more limited than the torch.einsum function
     * as it only supports two arguments
     */
    static TensorDual einsum(const std::string& arg, TensorDual& first, TensorDual& second) {

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find("->");
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 2);
        auto darg = arg1 + "z->" + arg2 + "z";

        //This is commutative so we can put the dual part at the end
        auto d1 = torch::einsum(darg, {first.r, second.d});
        auto d2 = torch::einsum(darg, {second.r, first.d});

        return TensorDual(std::move(r), std::move(d1 + d2));
    }



    static TensorDual einsum(const std::string& arg, const torch::Tensor& first, const TensorDual & second)
    {

        auto r = torch::einsum(arg, {first, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find("->");
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 2);
        auto darg = arg1 + "z->" + arg2 + "z";

        //This is commutative so we can put the dual part at the end
        auto d = torch::einsum(darg, {first, second.d});

        return TensorDual(r, d);
        
    }


    static TensorDual einsum(const std::string& arg, const TensorDual& first, const torch::Tensor & second)
    {
        auto r = torch::einsum(arg, {first.r, second});

        // Find the position of the '->' in the einsum string
        auto pos1 = arg.find(",");
        auto pos2 = arg.find("->");
        auto arg1 = arg.substr(0, pos1);
        auto arg2 = arg.substr(pos1+1,pos2-pos1-1);
        auto arg3 = arg.substr(pos2+2);
        auto darg = arg1 + "z," + arg2 + "->" + arg3 + "z";

        //This is commutative so we can put the dual part at the end
        auto d = torch::einsum(darg, {first.d, second});

        return TensorDual(r, d);

      return einsum(arg, second, first);        
    }


    static TensorDual einsum(const std::string& arg, std::vector<TensorDual> tensors) 
    {
        if (arg.find("->") == std::string::npos) {
            throw std::invalid_argument("first argument does not contain ->");
        }
        if (arg.find("z") != std::string::npos) {
            throw std::invalid_argument("z is a reserved character in einsum used to operate on dual numbers");
        }
        std::vector<torch::Tensor> r_tensors;
        std::vector<torch::Tensor> d_tensors;
        for (const auto& t : tensors) {
            r_tensors.push_back(t.r);
            d_tensors.push_back(t.d);
        }

        auto r = torch::einsum(arg, r_tensors);

        // Find the position of the '->' in the einsum string
        auto posa = arg.find("->");

        //Find the positions of the "," in the einsum string
        std::vector<size_t> poss{};
        size_t pos = arg.find(',');
    
        while(pos != std::string::npos) 
        {
           poss.push_back(pos);
           pos = arg.find(',', pos + 1);
        }
        //This works because the dual dimension is implied if absent
        
        torch::Tensor d = torch::zeros_like(tensors[0].d);
        for ( int i=0; i < tensors.size(); i++)
        {
            std::vector<torch::Tensor> r_tensorsc = r_tensors;
            r_tensorsc[i] = tensors[i].d;
            //insert a z right before the comma on the lhs
            std::string darg{};
            if ( i != tensors.size()-1) 
            {
                darg = arg.substr(0, poss[i]) + 
                               "z," + 
                               arg.substr(poss[i]+1, posa-poss[i]-1) + 
                               "->" + 
                               arg.substr(posa+2)+"z";
            }
            else
            {
                darg = arg.substr(0, posa) + 

                               "z->" + 
                               arg.substr(posa+2)+"z";
            }
            //std::cerr << "darg: " << darg << std::endl;
            d = d + torch::einsum(darg, r_tensorsc);
        }


        return TensorDual(r, d);
    }




    // Static member function to replicate the 'where' class method
    static TensorDual where(const torch::Tensor& cond, const TensorDual& x, const TensorDual& y) {
        torch::Tensor condr, condd;

        if (cond.dim() == 1) {
            condr = cond.unsqueeze(1).expand({-1, x.r.size(1)});
            condd = cond.unsqueeze(1).unsqueeze(2).expand({-1, x.d.size(1), x.d.size(2)});
        } else {
            condr = cond;
            condd = cond.unsqueeze(2).expand({-1, -1, x.d.size(2)});
        }

        auto xr = torch::where(condr, x.r, y.r);
        auto xd = torch::where(condd, x.d, y.d);

        return TensorDual(xr, xd);
    }


    // Static member function to replicate the 'sum' class method
    static TensorDual sum(const TensorDual& x) {
        // Compute sum along dimension 1 and unsqueeze
        auto r = torch::unsqueeze(torch::sum(x.r, /*dim=*/1), /*dim=*/1);
        auto d = torch::unsqueeze(torch::sum(x.d, /*dim=*/1), /*dim=*/1);

        // Return as TensorDual
        return TensorDual(r, d);
    }

    TensorDual normL2()
    {
       //Retain the dimensions of the tensors
       auto r_norm = torch::norm(this->r, 2, 1, true);
       auto r_norm_expanded = r_norm.expand_as(this->r);
       auto grad_r = r/r_norm;
       auto dual = torch::einsum("mi,mij->mj", {grad_r, this->d}).unsqueeze(1);
       return TensorDual(r_norm, dual);
    }





    TensorDual sum() {
        // Compute sum along dimension 1 and unsqueeze to retain the dimension
        auto real = torch::unsqueeze(torch::sum(r, /*dim=*/1), /*dim=*/1);
        auto dual = torch::unsqueeze(torch::sum(d, /*dim=*/1), /*dim=*/1);

        // Return as TensorDual
        return TensorDual(real, dual);
    }


    TensorDual clone() const {
        return TensorDual(r.clone(), d.clone());
    }

        // Overload the unary negation operator '-'
    TensorDual operator-() const {
        return TensorDual(-r, -d);
    }



        // Overload the addition operator for TensorDual
    TensorDual operator+(const TensorDual& other) const {
        return TensorDual(r + other.r, d + other.d);
    }

    // Overload the addition operator for torch::Tensor
    TensorDual operator+(const torch::Tensor& other) const {
        return TensorDual(std::move(r + other), std::move(d.clone()));
    }

    TensorDual operator+(double other) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({other}, this->r.options());
        return TensorDual(std::move(r + scalar_tensor), std::move(d.clone()));
    }


    // Overload the subtraction operator for TensorDual - TensorDual
    TensorDual operator-(const TensorDual& other) const {
        return TensorDual(r - other.r, d - other.d);
    }

    // Overload the subtraction operator for TensorDual - torch::Tensor
    // Assuming that 'other' can be broadcast to match the shape of 'r'
    TensorDual operator-(const torch::Tensor& other) const {
        return TensorDual(std::move(r - other), std::move(d.clone()));
    }

    TensorDual operator-(double&& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorDual(std::move(this->r - scalar_tensor), std::move(this->d.clone()));
    }

    TensorDual operator-(double& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorDual(std::move(this->r - scalar_tensor), std::move(this->d.clone()));
    }


    //add a method to pretty print the contents of the TensorDual object
    void print() const {
        std::cerr << "r: " << r << std::endl;
        std::cerr << "d: " << d << std::endl;
    }

    TensorDual contiguous() const {
        return TensorDual(r.contiguous(), d.contiguous());
    }



    // Assuming TensorDual is a defined class
    TensorDual operator*(const TensorDual& other) const {
        //std::cerr << "operator * with TensorDual called" << std::endl;
        // Check for correct dimensions if necessary
        auto real = other.r * r;
        auto dual = other.r.unsqueeze(-1) * d + r.unsqueeze(-1) * other.d;
        //std::cerr << "r sizes in *: " << r.sizes() << std::endl;
        //std::cerr << "d sizes in *: " << d.sizes() << std::endl;
        return TensorDual(real, dual);
    }


    // Assuming TensorDual is a defined class
    TensorDual operator*(const torch::Tensor& other) const {
        //std::cerr << "operator * with TensorDual called" << std::endl;
        // Check for correct dimensions if necessary
        auto real = other*r;
        auto dual = other.unsqueeze(-1) * d;
        //std::cerr << "r sizes in *: " << r.sizes() << std::endl;
        //std::cerr << "d sizes in *: " << d.sizes() << std::endl;
        return TensorDual(real, dual);
    }


    //overload the * operator for a scalar and a TensorDual
    TensorDual operator*(const double& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        //std::cerr << "scalar: " << scalar << std::endl;
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorDual(this->r * scalar_tensor, this->d * scalar_tensor);
    }


    //overload the comparison operators
    // Overload the less-than-or-equal-to operator for TensorDual <= TensorDual
    torch::Tensor operator<=(const TensorDual& other) const {
        auto mask = r <= other.r;
        return torch::squeeze(mask, 1);
    }

    // Overload the equals operator for TensorDual == TensorDual
    torch::Tensor operator==(const TensorDual& other) const {
        auto mask = r == other.r;
        return torch::squeeze(mask, 1);
    }


    // Overload the less-than-or-equal-to operator for TensorDual <= torch::Tensor
    // This also implicitly supports TensorDual <= Scalar due to tensor's implicit constructors
    torch::Tensor operator<=(const torch::Tensor& other) const {
        auto mask = r <= other;
        return torch::squeeze(mask, 1);
    }

    template <typename Scalar>
    torch::Tensor operator<=(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = (r <= scalar);
        return torch::squeeze(mask, 1);
    }

    //overload the > operator
    torch::Tensor operator>(const TensorDual& other) const {
        auto mask = r > other.r;
        return torch::squeeze(mask, 1);
    }

    //overload the > operator for a tensor and a TensorDual
    torch::Tensor operator>(const torch::Tensor& other) const {
        auto mask = r > other;
        return torch::squeeze(mask, 1);
    }

    //overload the > for a scalar and a TensorDual
    template <typename Scalar>
    torch::Tensor operator>(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        auto mask = r > scalar_tensor;
        return torch::squeeze(mask, 1);
    }

    //overload the < operator
    torch::Tensor operator<(const TensorDual& other) const {
        auto mask = r < other.r;
        return torch::squeeze(mask, 1);
    }

    //overload the < operator for a tensor and a TensorDual
    torch::Tensor operator<(const torch::Tensor& other) const {
        auto mask = r < other;
        //std::cerr << "mask sizes in <: " << mask.sizes() << std::endl;
        return torch::squeeze(mask, 1);
    }

    //overload the < for a scalar and a TensorDual
    template <typename Scalar>
    torch::Tensor operator<(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r < scalar;
        return torch::squeeze(mask, 1);
    }



    // Overload the greater-than-or-equal-to operator for TensorDual >= TensorDual
    torch::Tensor operator>=(const TensorDual& other) const {
        auto mask = r >= other.r;
        return torch::squeeze(mask, 1);
    }

    // Overload the greater-than-or-equal-to operator for TensorDual >= torch::Tensor
    // This also implicitly supports TensorDual >= Scalar due to tensor's implicit constructors
    torch::Tensor operator>=(const torch::Tensor& other) const {
        auto mask = r >= other;
        return torch::squeeze(mask, 1);
    }

    //overload the >= operator for TensorDual and a scalar
    template <typename Scalar>
    torch::Tensor operator>=(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        auto mask = r >= scalar_tensor;
        return torch::squeeze(mask, 1);
    }



    // Overload the equality operator for TensorDual == torch::Tensor
    // This also implicitly supports TensorDual == Scalar due to tensor's implicit constructors
    torch::Tensor operator==(const torch::Tensor& other) const {
        auto mask = r.eq(other); // Using the .eq() function for equality comparison
        return torch::squeeze(mask, 1);
    }

    template <typename Scalar>
    torch::Tensor operator==(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r.eq(scalar);
        return torch::squeeze(mask, 1);
    }

    // Overload the inequality operator for TensorDual != TensorDual
    torch::Tensor operator!=(const TensorDual& other) const {
        auto mask = r.ne(other.r); // Using the .ne() function for inequality comparison
        return torch::squeeze(mask, 1);
    }

    // Overload the inequality operator for TensorDual != torch::Tensor
    // This also implicitly supports TensorDual != Scalar due to tensor's implicit constructors
    torch::Tensor operator!=(const torch::Tensor& other) const {
        auto mask = r.ne(other); // Using the .ne() function for inequality comparison
        return torch::squeeze(mask, 1);
    }

    template <typename Scalar>
    torch::Tensor operator!=(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto mask = r.ne(scalar);
        return torch::squeeze(mask, 1);
    }





    // Member overload for TensorDual / TensorDual
    TensorDual operator/(const TensorDual& other) const {
        auto r = this->r / other.r;
        auto otherrsq = other.r.square();
        auto d = -(this->r / otherrsq).unsqueeze(-1) * other.d + this->d/other.r.unsqueeze(-1);
        //std::cerr << "d sizes in /: " << d.sizes() << std::endl;
        //std::cerr << "current dual sizes " << this->d_.sizes() << std::endl;
        //Make sure the dimensions stay the same as the input tensor
        return TensorDual(r, d);
    }

    TensorDual operator/(const torch::Tensor& other) const {
        auto othere = other.dim() != this->r.dim() ? other.unsqueeze(1) : other;
        auto r = this->r / othere;
        auto d = this->d / othere.unsqueeze(-1);
        return TensorDual(r, d);
    }

    TensorMatDual operator/(const TensorMatDual& other) const; // Forward declaration

    // Member overload for TensorDual / torch::Tensor
    // Assumes 'other' can be broadcasted to match the shape of 'this->r'
    /*TensorDual operator/(const torch::Tensor& other) const {
        auto otherd = TensorDual::toDual(other, *this);
        return *this / otherd;
    }*/


    //overload the / operator for a scalar and a TensorDual
    TensorDual operator/(double scalar)  {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor(scalar, this->r.options());
        return TensorDual(this->r / scalar_tensor, this->d / scalar_tensor);
    }



    // Method to gather elements along a given dimension
    TensorDual gather(int64_t dim, const torch::Tensor& index) const {
        // Unsqueeze index and expand it to match the shape of 'd'
        auto indexd = index.unsqueeze(2).expand({-1, -1, d.size(2)});
        // Gather on both the real and dual parts using the expanded index
        auto r_gathered = r.gather(dim, index);
        auto d_gathered = d.gather(dim, indexd);
        return TensorDual(r_gathered, d_gathered);
    }

    // Method to scatter elements along a given dimension
    TensorDual scatter(int64_t dim, const torch::Tensor& index, const TensorDual& src) const {
        // Unsqueeze index for the dual part to match the dimensions
        auto indexd = index.unsqueeze(-1);
        // Scatter on both the real and dual parts using the given indices and source
        auto r_scattered = r.scatter(dim, index, src.r);
        auto d_scattered = d.scatter(dim, indexd, src.d);
        return TensorDual(r_scattered, d_scattered);
    }


    // Method to compute the reciprocal of a TensorDual
    TensorDual reciprocal() const {
        auto rrec = r.reciprocal(); // Compute the reciprocal of the real part
        auto d = -rrec.unsqueeze(-1) * rrec.unsqueeze(-1) * this->d;
        return TensorDual(rrec, d);
    }

    TensorDual square() const {
        auto rsq = r.square(); // Compute the square of the real part
        auto d = 2 * r.unsqueeze(-1) * this->d;
        return TensorDual(rsq, d);
    }

    // Method to compute the sine of a TensorDual
    TensorDual sin() const {
        auto r = torch::sin(this->r); // Compute the sine of the real part
        auto s = torch::cos(this->r); // Compute the cosine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    // Method to compute the cosine of a TensorDual
    TensorDual cos() const {
        auto r = torch::cos(this->r); // Compute the cosine of the real part
        auto s = -torch::sin(this->r); // Compute the sine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    // Method to compute the tangent of a TensorDual
    TensorDual tan() const {
        auto r = torch::tan(this->r); // Compute the tangent of the real part
        auto s = torch::pow(torch::cos(this->r), -2); // Compute the secant as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the arcsine of a TensorDual
    TensorDual asin() const {
        auto r = torch::asin(this->r); // Compute the arcsine of the real part
        auto s = torch::pow(1 - torch::pow(this->r, 2), -0.5); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the arccosine of a TensorDual
    TensorDual acos() const {
        auto r = torch::acos(this->r); // Compute the arccosine of the real part
        auto s = -torch::pow(1 - torch::pow(this->r, 2), -0.5); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the arctangent of a TensorDual
    TensorDual atan() const {
        auto r = torch::atan(this->r); // Compute the arctangent of the real part
        auto s = torch::pow(1 + torch::pow(this->r, 2), -1); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the hyperbolic sine of a TensorDual
    TensorDual sinh() const {
        auto r = torch::sinh(this->r); // Compute the hyperbolic sine of the real part
        auto s = torch::cosh(this->r); // Compute the hyperbolic cosine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the hyperbolic cosine of a TensorDual
    TensorDual cosh() const {
        auto r = torch::cosh(this->r); // Compute the hyperbolic cosine of the real part
        auto s = torch::sinh(this->r); // Compute the hyperbolic sine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the hyperbolic tangent of a TensorDual
    TensorDual tanh() const {
        auto r = torch::tanh(this->r); // Compute the hyperbolic tangent of the real part
        auto s = torch::pow(torch::cosh(this->r), -2); // Compute the hyperbolic secant as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the hyperbolic arcsine of a TensorDual
    TensorDual asinh() const {
        auto r = torch::asinh(this->r); // Compute the hyperbolic arcsine of the real part
        auto s = torch::pow(1 + torch::pow(this->r, 2), -0.5); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }


    //Method to compute the hyperbolic arccosine of a TensorDual
    TensorDual acosh() const {
        //ensure all real elements are >=1
        assert((r >=1.0).all().item<bool>() && "All real elements passed to acosh must be >=1.0");
        auto r = torch::acosh(this->r); // Compute the hyperbolic arccosine of the real part
        auto s = torch::pow(torch::pow(this->r, 2.0) - 1.0, -0.5); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }


    //Method to compute the hyperbolic arctangent of a TensorDual
    TensorDual atanh() const {
        assert( (r<=1.0).all().item<bool>() && (r>=-1.0).all().item<bool>() && "All real values must be between -1.0 and 1.0 for atanh");
        auto r = torch::atanh(this->r); // Compute the hyperbolic arctangent of the real part
        auto s = torch::pow(1 - torch::pow(this->r, 2), -1); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the exponential of a TensorDual
    TensorDual exp() const {
        auto r = torch::exp(this->r); // Compute the exponential of the real part
        auto d = r.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the natural logarithm of a TensorDual
    TensorDual log() const {
        auto r = torch::log(this->r); // Compute the natural logarithm of the real part
        auto s = torch::pow(this->r, -1); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        return TensorDual(r, d);
    }

    //Method to compute the square root of a TensorDual
    //Method to compute the square root of a TensorDual
    TensorDual sqrt() const {
      // Compute the square root of the real part
      auto r = torch::sqrt(this->r);

      // Remove negative elements by setting them to zero
      auto rf = torch::where(r.is_complex() ? torch::real(r) > 0 : r > 0, r, torch::zeros_like(r));

      // Compute the dual part
      auto d = torch::einsum("mi, mij->mij", {0.5 / rf, this->d});

      return TensorDual(r, d);
    }
    //Method to compute the absolute value of a TensorDual
    TensorDual abs() const {
        auto abs_r = torch::abs(r); // Compute the absolute value of the real part
        auto sign_r = r.is_complex() ? torch::unsqueeze(torch::sign(torch::real(r)), -1) : torch::unsqueeze(torch::sign(r), -1); // Compute the sign and unsqueeze to match the dimensions of d
        auto abs_d = sign_r * d; // The dual part multiplies by the sign of the real part
        return TensorDual(abs_r, abs_d);
    }

    TensorDual sign() const {
        auto sign_r = torch::sign(r); // Compute the sign of the real part
        //base this purely on the real part
        auto sign_d = torch::zeros_like(d); // The dual part is zero
        return TensorDual(sign_r, sign_d);
    }

    TensorDual slog() const {
      auto r = torch::sign(this->r) * torch::log(torch::abs(this->r)+1.0); // Compute the sign and logarithm of the absolute value of the real part
      auto d = (torch::abs(this->r)+1.0).reciprocal().unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
      return TensorDual(r, d); 
    }

    TensorDual sloginv() const {
      auto r = torch::sign(this->r)*(torch::exp(torch::abs(this->r))-1.0); // Compute the sign and exponential of the absolute value of the real part
      auto d = torch::exp(torch::abs(this->r)).unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
      return TensorDual(r, d); 
    }

    TensorDual softsign() const {
      auto r = this->r / (1.0 + torch::abs(this->r)); // Compute the softsign of the real part
      auto d = (1.0 + torch::abs(this->r)).pow(-2).unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
      return TensorDual(r, d); 
    }

    TensorDual softsigninv() const {
      auto r = this->r / (1.0 - torch::abs(this->r)); // Compute the softsigninv of the real part
      auto d = (1.0 - torch::abs(this->r)).pow(-2).unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
      return TensorDual(r, d); 
    }

    TensorDual max() {
     // max_values, max_indices = torch.max(self.r, dim=1, keepdim=True)
     auto max_result = torch::max(this->r, /*dim=*/1, /*keepdim=*/true);    
     auto max_values = std::get<0>(max_result); // For the maximum values
     auto max_indices = std::get<1>(max_result); // For the indices of the maximum values

     //dshape = max_indices.unsqueeze(-1).expand(-1, -1, self.d.shape[-1])
     auto dshape = max_indices.unsqueeze(-1).expand({-1, -1, this->d.size(-1)});
     //dual_values = torch.gather(self.d, 1, dshape)
     auto dual_values = torch::gather(this->d, /*dim=*/1, dshape);
     //return TensorDual(max_values, dual_values)
     return TensorDual(max_values, dual_values);
    }

    TensorDual min() {
      // Compute the min values and indices along dimension 1, keeping the dimension
      auto min_result = torch::min(this->r, /*dim=*/1, /*keepdim=*/true);
      auto min_values = std::get<0>(min_result); // Minimum values
      auto min_indices = std::get<1>(min_result); // Indices of the minimum values

      // Adjust the shape of min_indices for gathering the dual values
      auto dshape = min_indices.unsqueeze(-1).expand({min_indices.size(0), min_indices.size(1), this->d.size(-1)});

      // Gather the dual values based on the min indices
      auto dual_values = torch::gather(this->d, /*dim=*/1, dshape);

      // Return a new TensorDual with the min values and corresponding dual values
      return TensorDual(min_values, dual_values);
    }

    TensorDual 
    complex() {
        torch::Tensor rc, dc;
        this->r.is_complex() ? rc = this->r : rc = torch::complex(this->r, torch::zeros_like(this->r)).to(this->r.device());
        this->d.is_complex() ? dc = this->d : dc = torch::complex(this->d, torch::zeros_like(this->d)).to(this->d.device());
        return TensorDual(std::move(rc), std::move(dc));
    }

    TensorDual real() {
        auto r = torch::real(this->r);
        auto d = torch::real(this->d);
        return TensorDual(std::move(r), std::move(d));
    }

    TensorDual imag() {
        auto r = torch::imag(this->r);
        auto d = torch::imag(this->d);
        return TensorDual(r, d);
    }
    

    
    TensorDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {
        auto r = this->r.index(indices);
        r.dim() == 1 ? r = r.unsqueeze(1) : r;
        auto d = this->d.index(indices);
        d.dim() == 2 ? d = d.unsqueeze(1) : d;
        return TensorDual(r, d);
    }



    TensorDual index(int index) {
        auto real = this->r.index({Slice(index, index+1)});
        auto dual = this->d.index({Slice(index, index+1)});
        return TensorDual(real, dual);
    }

    TensorDual index(const torch::Tensor& mask) {
        auto real = r.index({mask});
        auto dual = d.index({mask});
        return TensorDual(real, dual);
    }

    TensorDual index(const TensorIndex& index) {
        auto real = r.index({index});
        auto dual = d.index({index});
        return TensorDual(real, dual);
    }



    void index_put_(const torch::Tensor& mask, const TensorDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
    }

    /*void index_put_(const torch::Tensor& mask, const torch::Tensor& value) {
        auto valued = TensorDual::toDual(value, (*this).index({mask}));
        this->r.index_put_({mask}, valued.r);
        this->d.index_put_({mask}, valued.d);
    }*/

    /*void index_put_(const torch::Tensor& mask, const double& value) {
        auto valuet = torch::tensor({value}, this->r.options());
        auto valued = TensorDual::toDual(valuet, (*this).index({mask}));
        this->r.index_put_({mask}, valued.r);
        this->d.index_put_({mask}, valued.d);
    }*/


    void index_put_(const TensorIndex& mask, const TensorDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
    }

    template <typename Scalar>
    void index_put_(const TensorIndex& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const TensorDual& value) {
        this->r.index_put_(mask, value.r);
        this->d.index_put_(mask, value.d);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const double& value) {
        this->r.index_put_(mask, value);
        this->d.index_put_(mask, 0.0);
    }


    //Method to compute the max of a TensorDual
    TensorDual max(const TensorDual& other) const {
        auto r = torch::max(this->r, other.r); // Compute the max of the real part
        auto d = torch::where(this->r > other.r, this->d, other.d); // Compute the max of the dual part
        return TensorDual(r, d);
    }

    TensorDual sign()
    {
      auto r = torch::sign(this->r);
      auto d = torch::zeros_like(this->d);
      return TensorDual(r, d);
    }

    TensorDual pow(const double exponent) const {
        auto r = torch::pow(this->r, exponent); // Compute the power of the real part
        auto d = torch::einsum("mi, mij->mij",{exponent * torch::pow(this->r, exponent - 1) , 
                                               this->d}); // Compute the power of the dual part
        return TensorDual(r, d);
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

        // Precompute common terms
        auto r2sq = r2.square();                               // r2^2
        auto r2cube = r2 * r2sq;                               // r2^3
        auto r1d2 = r1.unsqueeze(-1) * d2;                    // r1 * d2
        auto r1h2 = r1.unsqueeze(-1).unsqueeze(-1) * h2;      // r1 * h2
        auto d1d2 = torch::einsum("...ij,...ij->...ij", d1, d2); // d1 * d2 (element-wise)
        auto r1d2d2 = torch::einsum("...i,...ij->...ij", r1, d2 * d2); // r1 * d2 * d2

        // Real part of the result
        auto rn = r1 / r2;

        // Dual part of the result
        auto dn = d1 / r2.unsqueeze(-1) - r1d2 / r2sq.unsqueeze(-1);

        // Hyperdual part of the result
        auto hn = h1 / r2.unsqueeze(-1).unsqueeze(-1) -
                2 * d1d2.unsqueeze(-1) / r2sq.unsqueeze(-1).unsqueeze(-1) -
                r1h2 / r2sq.unsqueeze(-1).unsqueeze(-1) +
                2 * r1d2d2.unsqueeze(-1) / r2cube.unsqueeze(-1).unsqueeze(-1);

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