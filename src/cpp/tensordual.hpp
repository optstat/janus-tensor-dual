#ifndef TENSORDUAL_H
#define TENSORDUAL_H

#include <torch/torch.h>
#include <type_traits> // For std::is_scalar
#include <vector>
using TensorIndex = torch::indexing::TensorIndex;
using Slice = torch::indexing::Slice;
namespace janus {

class TensorMatDual;

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
        auto r = this->r / other;
        auto d = this->d / other.unsqueeze(-1);
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
      auto d = torch::exp(this->r).unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
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
        auto d = torch::zeros({this->r.size(0), this->r.size(1), this->r.size(2), this->d.size(2)}, this->d.options());
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
        auto d = torch::einsum("mij, mijn->mijn", {0.5 / rf, this->d});
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
    
    
    static TensorMatDual cat(const TensorMatDual& t1, const TensorMatDual &t2)
    {
        auto r = torch::cat({t1.r, t2.r}, 2);
        auto d = torch::cat({t1.d, t2.d}, 2);
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


    //overload the - operator
    TensorMatDual operator-(const TensorMatDual& other) const {
        return TensorMatDual(this->r - other.r, this->d - other.d);
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
        return *this / other_mat;
    }

    TensorMatDual operator/(const torch::Tensor& other) const {
        auto real = this->r/other;
        auto dual = this->d/other.unsqueeze(-1);
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





// Non-member overload for torch::Tensor + TensorDual
TensorDual operator+(const torch::Tensor& tensor, const TensorDual& td) {
    return TensorDual(std::move(tensor + td.r), std::move(td.d.clone()));
}

// Non-member template function for Scalar + TensorDual
TensorDual operator+(const double& scalar, const TensorDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorDual((scalar + td.r).clone(), td.d.clone());
}

// Non-member overload for torch::Tensor - TensorDual
TensorDual operator-(const torch::Tensor& tensor, const TensorDual& td) {
    return TensorDual(std::move(tensor - td.r), std::move(-td.d.clone()));
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
    resr.index_put_({~mask}, rhs.index({~mask}));
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