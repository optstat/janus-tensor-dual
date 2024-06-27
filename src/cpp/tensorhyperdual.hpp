#ifndef TENSORHYPERDUAL_H
#define TENSORHYPERDUAL_H


#include <torch/torch.h>
#include <iostream>
#include "tensordual.hpp"
#include <type_traits> // For std::is_scalar

using TensorIndex = torch::indexing::TensorIndex;
namespace janus {
class TensorHyperDual {

public:
    // Members
    torch::Tensor r;
    torch::Tensor d;
    torch::Tensor hd;
    //Add a dtype
    torch::Dtype dtype;
    torch::Device device_= torch::kCPU;


public:
    /* 
    * Constructor for the TensorHyperDual class
    * @param r: The real part of the TensorHyperDual
    * @param d: The dual part of the TensorHyperDual
    * @param hd: The hyperdual part of the TensorHyperDual
    * For efficiency reasons, we assume that the input tensors are already on the correct device
    * and have the correct data type
    * There is no copy of the data. We just store the reference to the input tensors so one has to be 
    * careful when modifying the input tensors after creating a TensorHyperDual object
    * This definition is a simplification of the mathematical concept of hyperdual numbers
    * implemented for vectorized (data parallel) automatic differentiation
    */ 
    TensorHyperDual(torch::Tensor r, torch::Tensor d, torch::Tensor hd) {
        assert(r.dim()  == 2 && "The real part of the TensorHyperDual should be a matrix");
        assert(d.dim()  == 3 && "The dual part of the TensorHyperDual should be a 3D tensor");
        assert(hd.dim() == 4 && "The hyperdual part of the TensorHyperDual should be a 4D tensor");
        this->r = r;
        this->d = d;
        this->hd = hd;
        this->dtype = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }


    TensorHyperDual() {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

        // Create zero tensors with the specified options
        torch::Tensor rl{torch::zeros({1, 1}, options)};
        torch::Tensor dl{torch::zeros({1, 1, 1}, options)};
        TensorDual(rl, dl);

    }

    friend std::ostream& operator<<(std::ostream& os, const TensorHyperDual& obj){
        os << "r: " << obj.r << std::endl;
        os << "d: " << obj.d << std::endl;
        os << "hd: " << obj.hd << std::endl;
        return os;
    }

    torch::Device device() const {
        return device_;
    }

    void to_device(const torch::Device& device) {
        r = r.to(device);
        d = d.to(device);
        hd = hd.to(device);
        device_ = device;
    }





    // Constructor overload for scalar types
    template <typename S>
    TensorHyperDual(S r, S d, S hd, int64_t dim = 1) {
        auto options = torch::TensorOptions().dtype(torch::kFloat64); // You need to specify the correct data type here

        this->r = torch::tensor({r}, options);
        if (this->r.dim() == 1) {
            this->r = this->r.unsqueeze(dim);
        }

        this->d = torch::tensor({d}, options);
        if (this->d.dim() == 1) {
            this->d = this->d.unsqueeze(dim);
        }

        this -> hd = torch::tensor({hd}, options);
        if (this->hd.dim() == 1) {
            this->hd = this->hd.unsqueeze(dim);
        }
    }

    // Static member function to replicate the @classmethod 'create'
    static TensorHyperDual create(const torch::Tensor& r) {
        auto l = r.size(1); // number of leaves N
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(l); // add the extra dimension for the dual part
        auto hdshape = r.sizes().vec(); // copy the sizes to a vector
        hdshape.push_back(l); // add the extra dimension for the hyperdual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);

        // Set the last dimension as the identity matrix
        for (int64_t i = 0; i < l; ++i) {
            ds.index({torch::indexing::Slice(), i, i}) = 1.0;
        }

        auto hds = torch::zeros(dshape, options);
        for (int64_t i = 0; i < l; ++i) {
            hds.index({torch::indexing::Slice(), i, i}) = 1.0;
        }

        return TensorHyperDual(r, ds, hds);
    }

        // Static member function to create a TensorDual with zeros
    static TensorHyperDual zeros_like(const TensorHyperDual& x, torch::Dtype dtype = torch::kFloat64) {
        auto r = torch::zeros_like(x.r, dtype);
        auto ds = torch::zeros_like(x.d, dtype);
        auto hds = torch::zeros_like(x.hd, dtype);
        return TensorHyperDual(r, ds, hds);
    }

    // Static member function to create a TensorDual with ones
    static TensorHyperDual ones_like(const TensorHyperDual& x) {
        auto r = torch::ones_like(x.r, x.r.dtype());
        torch::Tensor ds, hds;
        if (r.dtype() == torch::kBool) {
            ds = torch::zeros_like(x.d, torch::kFloat64);
            hds = torch::zeros_like(x.hd, torch::kFloat64);
        } else {
            ds = torch::zeros_like(x.d, x.r.dtype());
            hds = torch::zeros_like(x.hd, x.r.dtype());
        }
        return TensorHyperDual(r, ds, hds);
    }

     // Static member function to create a boolean tensor like the real part of x
    static torch::Tensor bool_like(const TensorHyperDual& x) {
        return torch::zeros_like(x.r, torch::dtype(torch::kBool).device(x.r.device()));
    }

    static TensorHyperDual createZero(const torch::Tensor& r, int ddim) {
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(ddim); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        auto hds = torch::zeros(dshape, options);
        return TensorHyperDual(r, ds, hds);
    }

        // Static member function to concatenate TensorDual objects
    static TensorHyperDual cat(const std::vector<TensorHyperDual>& args) {
        // Check to make sure all elements of args are TensorDual
        std::vector<torch::Tensor> r_tensors;
        std::vector<torch::Tensor> d_tensors;
        std::vector<torch::Tensor> hd_tensors;

        for (const auto& a : args) {
            r_tensors.push_back(a.r);
            d_tensors.push_back(a.d);
            hd_tensors.push_back(a.hd);
        }

        auto r = torch::squeeze(torch::stack(r_tensors, 1), 2);
        auto d = torch::squeeze(torch::stack(d_tensors, 1), 2);
        auto hd = torch::squeeze(torch::stack(hd_tensors, 1), 2);

        return TensorHyperDual(r, d, hd);
    }


    // Static member function to replicate the 'where' class method
    static TensorHyperDual where(const torch::Tensor& cond, const TensorHyperDual& x, const TensorHyperDual& y) {
        torch::Tensor condr, condd, condhd;

        if (cond.dim() == 1) {
            condr = cond.unsqueeze(1).expand({-1, x.r.size(1)});
            condd = cond.unsqueeze(1).unsqueeze(2).expand({-1, x.d.size(1), x.d.size(2)});
            condhd = cond.unsqueeze(1).unsqueeze(2).expand({-1, x.hd.size(1), x.hd.size(2)});
        } else {
            condr = cond;
            condd = cond.unsqueeze(2).expand({-1, -1, x.d.size(2)});
            condhd = cond.unsqueeze(2).expand({-1, -1, x.hd.size(2)});
        }

        auto xr = torch::where(condr, x.r, y.r);
        auto xd = torch::where(condd, x.d, y.d);
        auto xhd = torch::where(condhd, x.hd, y.hd);

        return TensorHyperDual(xr, xd, xhd);
    }


    /*
     * Static member function to replicate the 'sum' class method
     * Note this is only implemented for the case where dim=1
     */

    static TensorHyperDual sum(const TensorHyperDual& x) {
        // Compute sum along dimension 1 and unsqueeze
        auto r = torch::unsqueeze(torch::sum(x.r, /*dim=*/1), /*dim=*/1);
        auto d = torch::unsqueeze(torch::sum(x.d, /*dim=*/1), /*dim=*/1);
        auto hd = torch::unsqueeze(torch::sum(x.hd, /*dim=*/1), /*dim=*/1);
        // Return as TensorDual
        return TensorHyperDual(r, d, hd);
    }




    /*
     * Static member function to replicate the 'sum' class method
     * Note this is only implemented for the case where dim=1
     */

    TensorHyperDual sum() {
        // Compute sum along dimension 1 and unsqueeze
        auto real = torch::unsqueeze(torch::sum(r, /*dim=*/1), /*dim=*/1);
        auto dual = torch::unsqueeze(torch::sum(d, /*dim=*/1), /*dim=*/1);
        auto hyperdual = torch::unsqueeze(torch::sum(hd, /*dim=*/1), /*dim=*/1);

        // Return as TensorDual
        return TensorHyperDual(real, dual, hyperdual);
    }


    TensorHyperDual clone() const {
        return TensorHyperDual(r.clone(), d.clone(), hd.clone());
    }

        // Overload the unary negation operator '-'
    TensorHyperDual operator-() const {
        return TensorHyperDual(-r, -d, -hd);
    }


    //overload the - operator for a scalar and a TensorDual
    template <typename Scalar>
    TensorHyperDual operator-(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorHyperDual(this->r - scalar_tensor, this->d, this->hd);
    }


        // Overload the addition operator for TensorDual
    TensorHyperDual operator+(const TensorHyperDual& other) const {
        return TensorHyperDual(r + other.r, d + other.d, hd + other.hd);
    }

    // Overload the addition operator for torch::Tensor
    TensorHyperDual operator+(const torch::Tensor& other) const {
        auto otherr = other.reshape_as(r);
        return TensorHyperDual(r + otherr, d, hd);
    }


    // Overload the subtraction operator for TensorDual - TensorDual
    TensorHyperDual operator-(const TensorHyperDual& other) const {
        return TensorHyperDual(r - other.r, d - other.d, hd - other.hd);
    }

    // Overload the subtraction operator for TensorDual - torch::Tensor
    // Assuming that 'other' can be broadcast to match the shape of 'r'
    TensorHyperDual operator-(const torch::Tensor& other) const {
        return TensorHyperDual(r - other, d, hd);
    }

    //add a method to pretty print the contents of the TensorDual object
    void print() const {
        std::cerr << "r: " << r << std::endl;
        std::cerr << "d: " << d << std::endl;
        std::cerr << "hd: " << hd << std::endl;
    }


    // Helper function to convert a scalar or a torch::Tensor to a TensorHyperDual
    static TensorHyperDual toHyperDual(const torch::Tensor& other, const torch::Tensor& reference) {
        return TensorHyperDual(other.expand_as(reference), torch::zeros_like(reference), torch::zeros_like(reference));
    }

    TensorHyperDual reciprocal() const 
    {
        auto rrec = r.reciprocal(); // Compute the reciprocal of the real part
        auto rrecsq = rrec.square();
        auto rrec3 = rrec.pow(3);
        auto d = -torch::einsum("mi, mij->mij",  {-rrecsq,this->d});
        //This is the hessian of the reciprocal function
        auto outer = torch::einsum("mij,mik->mijk", {this->d, this->d});
        auto hd = 2*torch::einsum("mi,mijk->mijk", {rrec3, outer})-
                   torch::einsum("mi,mijk->mijk", {rrecsq, this->hd});
        return TensorHyperDual(rrec, d, hd);
    }
    
    /**
     * Assuming TensorHyperDual is a defined class
     * This is simply elementwise multiplication of the real parts
     * and corresponding dual parts 
     */
    TensorHyperDual operator*(const TensorHyperDual& other) const { 
        //std::cerr << "operator * with TensorDual called" << std::endl;
        // Check for correct dimensions if necessary
        auto real = other.r * r; //This is done elementwise
        auto dual = torch::einsum("mi, mij->mij",{this->r, other.d})+
                    torch::einsum("mij, mi->mij",{this->d, other.r});         
        //https://math.stackexchange.com/questions/129424/double-derivative-of-the-composite-of-functions
        auto hyperdual = torch::einsum("mi, mijk->mijk",{this->r, other.hd})+
                         2*torch::einsum("mij, mik->mijk",{this->d, other.d})+
                         torch::einsum("mijk, mi->mijk",{this->hd, other.r});
        return TensorHyperDual(real, dual, hyperdual);
    }

    //overload the * operator for a scalar and a TensorDual
    TensorHyperDual operator*(const double& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        //std::cerr << "scalar: " << scalar << std::endl;
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorHyperDual(this->r * scalar_tensor, this->d * scalar_tensor, this->hd * scalar_tensor);
    }


    //overload the comparison operators
    // Overload the less-than-or-equal-to operator for TensorDual <= TensorDual
    torch::Tensor operator<=(const TensorHyperDual& other) const {
        auto mask = r <= other.r;
        return torch::squeeze(mask, 1);
    }

    //overload the comparison operators
    // Overload the less-than-or-equal-to operator for TensorDual <= TensorDual
    torch::Tensor operator<=(const TensorDual& other) const {
        auto mask = r <= other.r;
        return torch::squeeze(mask, 1);
    }

    // Overload the equals operator for TensorDual == TensorDual
    torch::Tensor operator==(const TensorHyperDual& other) const {
        auto mask = r == other.r;
        return torch::squeeze(mask, 1);
    }

    // Overload the equals operator for TensorDual == TensorDual
    torch::Tensor operator==(const TensorDual& other) const {
        auto mask = r == other.r;
        return torch::squeeze(mask, 1);
    }

    // Overload the equals operator for TensorDual == TensorDual
    template <typename Scalar>
    torch::Tensor operator==(const Scalar& other) const {
        auto mask = r == other;
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
    TensorHyperDual operator/(const TensorHyperDual& other) const {
      auto real = this->r/other.r;
      auto dual = (this->d/other.r.unsqueeze(-1)) - 
                   (this->r.unsqueeze(-1)*other.d/other.r.unsqueeze(-1).unsqueeze(-1));
      auto hyperdual = (this->hd/other.r.unsqueeze(-1).unsqueeze(-1)) - 
                        (this->d/other.r.unsqueeze(-1).unsqueeze(-1).square())*other.d.unsqueeze(-1) +
                        this->d.unsqueeze(-1)*other.d.unsqueeze(-1)/other.r.unsqueeze(-1).unsqueeze(-1).square() -
                        2*this->r.unsqueeze(-1).unsqueeze(-1)*other.d.unsqueeze(-1).pow(3)/other.r.unsqueeze(-1).unsqueeze(-1).pow(2) +
                        this->r.unsqueeze(-1).unsqueeze(-1)*other.hd/other.r.unsqueeze(-1).unsqueeze(-1).square();
      return TensorHyperDual(real, dual, hyperdual);  
    }

    // Member overload for TensorDual / torch::Tensor
    // Assumes 'other' can be broadcasted to match the shape of 'this->r'
    TensorHyperDual operator/(const torch::Tensor& other) const {
        auto real = this->r/other;
        auto dual = this->d/other;
        auto hyperdual = this->hd/other;
        return TensorHyperDual(real, dual, hyperdual);
    }


    //overload the / operator for a scalar and a TensorDual
    template <typename Scalar>
    TensorHyperDual operator/(const Scalar& scalar) const {
        // Ensure the scalar is of a type convertible to Tensor
        auto scalar_tensor = torch::tensor({scalar}, this->r.options());
        return TensorHyperDual(this->r / scalar_tensor, 
                               this->d / scalar_tensor, 
                               this->hd / scalar_tensor);
    }




    // Overload the unary plus operator to represent the absolute value
    TensorDual operator+() const {
        auto abs_r = torch::abs(r); // Compute the absolute value of the real part
        auto sign_r = torch::unsqueeze(torch::sign(r), -1); // Compute the sign and unsqueeze to match the dimensions of d
        auto abs_d = sign_r * d; // The dual part multiplies by the sign of the real part
        return TensorDual(abs_r, abs_d);
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



    TensorHyperDual square() const {
        auto rsq = r.square(); // Compute the square of the real part
        auto dr = 2 * r.unsqueeze(-1);
        auto dual = dr * this->d; // Apply the scaling factor to the dual part
        auto hyperdual = 2 *this->hd;
        return TensorHyperDual(rsq, dual, hyperdual);
    }

    // Method to compute the sin of a TensorHyperDual
    TensorHyperDual sin() const {
        auto r = torch::sin(this->r); // Compute the sine of the real part
        auto dr = torch::cos(this->r); // Compute the cosine as the scaling factor for the dual part
        auto d = dr.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto d2r = -torch::sin(this->r); // Compute the negative sine as the scaling factor for the hyperdual part
        auto hd = d2r.unsqueeze(-1).unsqueeze(-1) * this->hd;
        return TensorHyperDual(r, d, hd);
    }

    // Method to compute the cosine of a TensorDual
    TensorHyperDual cos() const {
        auto r = torch::cos(this->r); // Compute the cosine of the real part
        auto dr = -torch::sin(this->r); // Compute the sine as the scaling factor for the dual part
        auto d = dr.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto d2r = -torch::cos(this->r); // Compute the negative cosine as the scaling factor for the hyperdual part
        auto hd =  d2r.unsqueeze(-1).unsqueeze(-1)* this->hd;
        return TensorHyperDual(r, d, hd);
    }

    // Method to compute the tangent of a TensorDual
    TensorHyperDual tan() const {
        auto r = torch::tan(this->r); // Compute the tangent of the real part
        auto s = torch::pow(torch::cos(this->r), -2); // Compute the secant as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = 2 * torch::pow(torch::cos(this->r), -3) * this->d.square() + torch::pow(torch::cos(this->r), -2) * this->hd;
        return TensorHyperDual(r, d, hd);
    }


    //Method to compute the hyperbolic sine of a TensorDual
    TensorHyperDual sinh() const {
        auto r = torch::sinh(this->r); // Compute the hyperbolic sine of the real part
        auto s = torch::cosh(this->r); // Compute the hyperbolic cosine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = torch::sinh(this->r).unsqueeze(-1) * this->d.square() + torch::cosh(this->r).unsqueeze(-1) * this->hd;
        return TensorHyperDual(r, d, hd);
    }

    //Method to compute the hyperbolic cosine of a TensorDual
    TensorHyperDual cosh() const {
        auto r = torch::cosh(this->r); // Compute the hyperbolic cosine of the real part
        auto s = torch::sinh(this->r); // Compute the hyperbolic sine as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = torch::cosh(this->r).unsqueeze(-1) * this->d.square() + torch::sinh(this->r).unsqueeze(-1) * this->hd;
        return TensorHyperDual(r, d, hd);
    }

    //Method to compute the hyperbolic tangent of a TensorDual
    TensorHyperDual tanh() const {
        auto r = torch::tanh(this->r); // Compute the hyperbolic tangent of the real part
        auto s = torch::pow(torch::cosh(this->r), -2); // Compute the hyperbolic secant as the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = 2 * torch::pow(torch::cosh(this->r), -3) * this->d.square() - torch::pow(torch::cosh(this->r), -2) * this->hd;
        return TensorHyperDual(r, d, hd);
    }


    //Method to compute the exponential of a TensorDual
    TensorHyperDual exp() const {
        auto r = torch::exp(this->r); // Compute the exponential of the real part
        auto d = r.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = r.unsqueeze(-1) * this->hd;
        return TensorHyperDual(r, d, hd);
    }

    //Method to compute the natural logarithm of a TensorDual
    TensorHyperDual log() const {
        auto r = torch::log(this->r); // Compute the natural logarithm of the real part
        auto s = torch::pow(this->r, -1); // Compute the scaling factor for the dual part
        auto d = s.unsqueeze(-1) * this->d; // Apply the scaling factor to the dual part
        auto hd = -torch::pow(this->r, -2).unsqueeze(-1) * this->d.square() + s.unsqueeze(-1) * this->hd;
        return TensorHyperDual(r, d, hd);
    }

    //Method to compute the square root of a TensorDual
    //Method to compute the square root of a TensorDual
    TensorHyperDual sqrt() const {
        auto r = torch::sqrt(this->r); // Compute the square root of the real part
        //rf = torch.where(r > 0, r, torch.zeros_like(r)) #Remove negative elements
        auto rf = torch::where(r > 0, r, torch::zeros_like(r)); // Remove negative elements
        //d = torch.einsum('mi, mij->mij', 0.5/rf, self.d)
        auto d = 0.5/rf.unsqueeze(-1) * this->d;
        auto hd = -(1.0/4.0)*torch::pow(this->r.unsqueeze(-1), -3/2)*this->d.square() + 0.5*torch::pow(this->r.unsqueeze(-1), -1/2)*this->hd;
        return TensorHyperDual(r, d, hd);
    }

    //Method to compute the absolute value of a TensorDual
    TensorHyperDual abs() const {
        auto abs_r = torch::abs(r); // Compute the absolute value of the real part
        auto sign_r = torch::unsqueeze(torch::sign(r), -1); // Compute the sign and unsqueeze to match the dimensions of d
        auto abs_d = sign_r * d; // The dual part multiplies by the sign of the real part
        auto abs_hd = sign_r * hd;
        return TensorHyperDual(abs_r, abs_d, abs_hd);
    }

    TensorHyperDual max() {
         // max_values, max_indices = torch.max(self.r, dim=1, keepdim=True)
         auto max_result = torch::max(this->r, /*dim=*/1, /*keepdim=*/true);
         auto max_values = std::get<0>(max_result); // For the maximum values
         auto max_indices = std::get<1>(max_result); // For the indices of the maximum values

        //dshape = max_indices.unsqueeze(-1).expand(-1, -1, self.d.shape[-1])
        auto dshape = max_indices.unsqueeze(-1).expand({-1, -1, this->d.size(-1)});
        //dual_values = torch.gather(self.d, 1, dshape)
        auto dual_values = torch::gather(this->d, /*dim=*/1, dshape);
        auto hyper_dual_values = torch::gather(this->hd, /*dim=*/1, dshape);
        //return TensorDual(max_values, dual_values)
        return TensorHyperDual(max_values, dual_values, hyper_dual_values);
    }


    TensorHyperDual index(int index) {
        auto real = this->r.index({index});
        auto dual = this->d.index({index});
        auto hyperdual = this->hd.index({index});
        return TensorHyperDual(real, dual, hyperdual);
    }

    TensorHyperDual index(const torch::Tensor& mask) {
        auto real = r.index({mask});
        auto dual = d.index({mask});
        auto hyperdual = hd.index({mask});
        return TensorHyperDual(real, dual, hyperdual);
    }

    TensorHyperDual index(const TensorIndex& index) {
        auto real = r.index(index);
        auto dual = d.index(index);
        auto hyperdual = hd.index(index);
        return TensorHyperDual(real, dual, hyperdual);
    }

    TensorHyperDual index(const std::vector<TensorIndex>& index) {
        auto real = r.index(index);
        auto dual = d.index(index);
        auto hyperdual = hd.index(index);
        return TensorHyperDual(real, dual, hyperdual);
    }


    void index_put_(const torch::Tensor& mask, const TensorHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const torch::Tensor& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
        this->hd.index_put_({mask}, 0.0);
    }

    void index_put_(const TensorIndex& mask, const TensorHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const TensorIndex& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
        this->hd.index_put_({mask}, 0.0);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const TensorHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const std::vector<TensorIndex>& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
    }


    //Method to compute the max of a TensorHyperDual
    TensorHyperDual max(const TensorHyperDual& other) const {
        auto r = torch::max(this->r, other.r); // Compute the max of the real part
        auto d = torch::where(this->r > other.r, this->d, other.d); // Compute the max of the dual part
        auto hd = torch::where(this->r > other.r, this->hd, other.hd); // Compute the max of the hyperdual part
        return TensorHyperDual(r, d, hd);
    }
};

class TensorMatHyperDual {
public:
    torch::Tensor r;
    torch::Tensor d;
    torch::Tensor hd;
    torch::Device device_ = torch::kCPU;

public:


    // Constructor
    TensorMatHyperDual(torch::Tensor r, torch::Tensor d, torch::Tensor hd) {
        this->r = r;
        this->d = d;
        this->hd = hd;
        this->device_ = r.device();
    }

    // Add constructor for a TensorMatDual from a TensorMatDual
    TensorMatHyperDual(const TensorMatHyperDual& other) {
        this->r = other.r;
        this->d = other.d;
        this->hd = other.hd;
        this->device_ = other.device_;
    }

    torch::Device device() const {
        return this->device_;
    }




    TensorMatHyperDual() {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU); // You need to specify the correct data type here

        // Create zero tensors with the specified options
        torch::Tensor rl{torch::zeros({1, 1, 1}, options)};
        torch::Tensor dl{torch::zeros({1, 1, 1, 1}, options)};
        torch::Tensor hdl{torch::zeros({1, 1, 1, 1, 1}, options)};
        TensorMatHyperDual(rl, dl, hdl);

    }


    friend std::ostream& operator<<(std::ostream& os, const TensorMatHyperDual& obj){
        os << "r: " << obj.r << std::endl;
        os << "d: " << obj.d << std::endl;
        os << "hd: " << obj.hd << std::endl;
        return os;
    }


    // Constructor overload for scalar types
    template <typename S>
    TensorMatHyperDual(S r, S d, S hd, int64_t dim = 1) {
        auto options = torch::TensorOptions().dtype(torch::kFloat64); // You need to specify the correct data type here

        this->r = torch::tensor({r}, options);
        if (this->r.dim() == 1) {
            this->r = this->r.unsqueeze(dim);
        }

        this->d = torch::tensor({d}, options);
        if (this->d.dim() == 1) {
            this->d = this->d.unsqueeze(dim);
        }
        this->hd = torch::tensor({hd}, options);
        if (this->hd.dim() == 1) {
            this->hd = this->hd.unsqueeze(dim);
        }
    }

    static TensorMatHyperDual createZero(const torch::Tensor& r, int ddim) {
        auto dshape = r.sizes().vec(); // copy the sizes to a vector
        dshape.push_back(ddim); // add the extra dimension for the dual part

        // Create a zero tensor for the dual part with the new shape
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);
        auto hds = torch::zeros(dshape, options);
        return TensorMatHyperDual(r, ds, hds);
    }


    TensorMatHyperDual clone() const {
        return TensorMatHyperDual(this->r.clone(), this->d.clone(), this->hd.clone());
    }


    //overload the + operator
    TensorMatHyperDual operator+(const TensorMatHyperDual& other) const {
        return TensorMatHyperDual(this->r + other.r, this->d + other.d, this->hd + other.hd);
    }

    //overload the + operator for the TensorMatDual and a TensorDual
    TensorMatHyperDual operator+(const TensorHyperDual& other) const {
        //std::cerr << "operator + with TensorDual called" << std::endl;
        auto r = this->r + other.r.unsqueeze(-1);
        //std::cerr << "r sizes in +: " << r.sizes() << std::endl;
        //std::cerr << "d sizes in +: " << this->d.sizes() << std::endl;
        //std::cerr << "other.d sizes in +: " << other.d.unsqueeze(-2).sizes() << std::endl;
        auto d = this->d + other.d.unsqueeze(-2);
        auto hd = this->hd + other.hd.unsqueeze(-2);
        //std::cerr << "d sizes in +: " << d.sizes() << std::endl;
        return TensorMatHyperDual(r, d, hd);
    }

    //overload the - operator
    TensorMatHyperDual operator-(const TensorMatHyperDual& other) const {
        return TensorMatHyperDual(this->r - other.r, this->d - other.d, this->hd - other.hd);
    }


    // Overload the equals operator for TensorDual == TensorDual
    torch::Tensor operator==(const TensorMatHyperDual& other) const {
        auto mask = r == other.r;
        return torch::squeeze(mask, 2);
    }



    //overload the - operator
    TensorMatHyperDual operator-() const {
        return TensorMatHyperDual(-this->r, -this->d, -this->hd);
    }


    TensorMatHyperDual index(const torch::Tensor& mask) {
        auto real = r.index({mask});
        auto dual = d.index({mask});
        auto hyperdual = hd.index({mask});
        return TensorMatHyperDual(real, dual, hyperdual);
    }

    TensorMatHyperDual index(const TensorIndex& index) {
        auto real = r.index(index);
        auto dual = d.index(index);
        auto hyperdual = hd.index(index);
        return TensorMatHyperDual(real, dual, hyperdual);
    }

    TensorMatHyperDual index(const std::vector<TensorIndex>& index) {
        auto real = r.index(index);
        auto dual = d.index(index);
        auto hyperdual = hd.index(index);
        return TensorMatHyperDual(real, dual, hyperdual);
    }


    void index_put_(const torch::Tensor& mask, const TensorHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const torch::Tensor& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
        this->hd.index_put_({mask}, 0.0);
    }

    void index_put_(const TensorIndex& mask, const TensorHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const TensorIndex& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
        this->hd.index_put_({mask}, 0.0);
    }

    void index_put_(const std::vector<TensorIndex>& mask, const TensorMatHyperDual& value) {
        this->r.index_put_({mask}, value.r);
        this->d.index_put_({mask}, value.d);
        this->hd.index_put_({mask}, value.hd);
    }

    template <typename Scalar>
    void index_put_(const std::vector<TensorIndex>& mask, const Scalar& value) {
        this->r.index_put_({mask}, value);
        this->d.index_put_({mask}, 0.0);
        this->hd.index_put_({mask}, 0.0);
    }

};






// Non-member overload for torch::Tensor + TensorDual
TensorHyperDual operator+(const torch::Tensor& tensor, const TensorHyperDual& td) {
    return TensorHyperDual(tensor + td.r, td.d, td.hd);
}

// Non-member template function for Scalar + TensorDual
template <typename Scalar>
TensorHyperDual operator+(const Scalar& scalar, const TensorHyperDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    auto scalar_tensor = torch::tensor({scalar}, td.r.options());
    return TensorHyperDual(scalar_tensor + td.r, td.d, td.hd);
}

// Non-member overload for torch::Tensor + TensorDual
TensorHyperDual operator-(const torch::Tensor& tensor, const TensorHyperDual& td) {
    return TensorHyperDual(tensor - td.r, td.d, td.hd);
}



TensorHyperDual operator*(const TensorHyperDual& td, const TensorMatHyperDual& other)  {
        auto r = torch::einsum("mi, mij->mj",{td.r,other.r});
        auto d = torch::einsum("mjn, mji->min", {td.d, other.r}) +
                 torch::einsum("mi, mijn->mjn", {td.r, other.d});
        auto hd = torch::einsum("mjn, mji->min", {td.hd, other.r}) +
                  2*torch::einsum("mi, mijn->mjn", {td.d, other.d}) +
                  torch::einsum("mi, mij->mj", {td.r, other.hd});
        return TensorHyperDual(r, d, hd);
}

TensorHyperDual operator*(const TensorMatHyperDual& tmd, const TensorHyperDual& other)  {
        auto r = torch::einsum("mij, mj->mi",{tmd.r, other.r});
        auto d = torch::einsum("mijn, mj->min", {tmd.d, other.r}) +
                 torch::einsum("mij, mjn->min", {tmd.r, other.d});
        auto hd = torch::einsum("mijn, mj->min", {tmd.hd, other.r}) +
                  2*torch::einsum("mij, mjn->min", {tmd.d, other.d}) +
                  torch::einsum("mij, mj->mi", {tmd.r, other.hd});
        return TensorHyperDual(r, d, hd);
}

//create a scalar overload for *
template <typename Scalar>
TensorDual operator*(const Scalar& scalar, const TensorHyperDual& td) {
    // Ensure the scalar is of a type convertible to Tensor
    return TensorDual(td.r * scalar, td.d * scalar, td.hd * scalar);
}




// Non-member function to handle scalar - TensorDual
TensorHyperDual operator-(const double& scalar, const TensorHyperDual& td) {
    auto scalar_tensor = torch::tensor({scalar}, td.r.options()); // Create a tensor filled with 'scalar'
    return TensorHyperDual(scalar_tensor - td.r, -td.d, -td.hd);
}

// Non-member function to handle scalar - TensorDual
TensorHyperDual operator-(const int& scalar, const TensorHyperDual& td) {
    auto scalar_tensor = torch::tensor({scalar}, td.r.options()); // Create a tensor filled with 'scalar'
    return TensorHyperDual(scalar_tensor - td.r, -td.d, -td.hd);
}



// Non-member template function for Scalar / TensorDual
template <typename Scalar>
TensorHyperDual operator/(const Scalar& scalar, const TensorHyperDual& td) {
    auto r = scalar/td.r;
    auto d = -(scalar/(td.r.square())).unsqueeze(-1) * td.d;
    auto hd = 2*(scalar/(td.r.square())).unsqueeze(-1) * td.d - (scalar/(td.r.square())).unsqueeze(-1) * td.hd;
    return TensorHyperDual(r, d, hd);
}



//overload the * operator for a tensor and a TensorDual
TensorHyperDual operator*(const torch::Tensor& tensor, const TensorHyperDual& td) {
    // Check for correct dimensions if necessary
    auto r = tensor * td.r;
    auto d = tensor * td.d;
    auto hd = tensor * td.hd;
    return TensorHyperDual(r, d, hd);
}

//overload the * operator for a double and a TensorDual
TensorHyperDual operator*(const double& scalar, const TensorHyperDual& td) 
{
    // Ensure the scalar is of a type convertible to Tensor
    return TensorHyperDual(td.r * scalar, td.d * scalar, td.hd * scalar);
}


static TensorHyperDual einsum(const std::string& arg, const TensorHyperDual& first, const torch::Tensor& second) 
{

        auto r = torch::einsum(arg, {first.r, second});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        auto d1 = torch::einsum(darg1, {first.d,  second});
        auto darg2 = arg1+","+arg2+"z->"+arg3+"z";
        auto d2 = torch::einsum(darg2, {first.r, second});

        auto hdarg = arg1 + "zZ," + arg2 + "->" + arg3 + "zZ";

        auto hd = torch::einsum(hdarg, {first.hd, second});

        return TensorHyperDual(std::move(r), std::move(d1 + d2), std::move(hd));
}

static TensorHyperDual einsum(const std::string& arg, const TensorHyperDual& first, const TensorHyperDual& second) 
{
    //TODO Add checks for the dimensions of the tensors

        auto r = torch::einsum(arg, {first.r, second.r});

        // Find the position of the '->' in the einsum string
        auto pos = arg.find(",");
        auto arg1 = arg.substr(0, pos);
        int pos2 = arg.find("->");
        auto arg2 = arg.substr(pos + 1, pos2-pos-1);
        auto arg3 = arg.substr(pos2 + 2);

        auto darg = arg1 + "z," + arg2 + "Z->" + arg3 + "zZ";
        auto d1 = torch::einsum(darg, {first.d,  second.r});
        auto darg2 = arg1+"z,"+arg2+"Z->"+arg3+"zZ";
        auto d2 = torch::einsum(darg2, {first.r, second.d});


        auto hdarg1 = arg1 + "zZ," + arg2 + "->" + arg3 + "zZ";
        auto hdarg2 = arg1 + "," + arg2 + "zZ->" + arg3 + "zZ";

        auto hd = torch::einsum(hdarg1, {first.hd, second.d}) + torch::einsum(darg2, {first.d, second.hd});

        return TensorHyperDual(std::move(r), std::move(d1 + d2), std::move(hd));
}


//pow method
TensorHyperDual pow(const TensorHyperDual& base, const torch::Tensor& exponent)  {
        auto r = torch::pow(base.r, exponent); // Compute the power of the real part
        //dual = torch.einsum('mi, mij->mij',other * self.r ** (other - 1), self.d)
        auto d = torch::einsum("mi, mij->mij", {exponent * torch::pow(base.r, exponent - 1), base.d});
        auto hd = torch::einsum("mi, mij->mij", {exponent * torch::pow(base.r, exponent - 1), base.hd}) +
                  torch::einsum("mi, mij->mij", {torch::pow(base.r, exponent) * torch::log(base.r), base.d});
        return TensorHyperDual(r, d, hd);
}


//overload the pow method for a TensorDual and a scalar
template <typename Scalar>
TensorHyperDual pow(const TensorHyperDual&base, const Scalar& exponent) {
    auto real = torch::pow(base.r, exponent);
    auto dual = torch::einsum("mi, mij->mij", {exponent * torch::pow(base.r, exponent - 1), base.d});
    auto hd = torch::einsum("mi, mij->mij", {exponent * torch::pow(base.r, exponent - 1), base.hd}) +
              torch::einsum("mi, mij->mij", {torch::pow(base.r, exponent) * torch::log(base.r), base.d});
    //std::cerr << "dual sizes in ^ with scalar: " << dual.sizes() << std::endl;
    return TensorHyperDual(real, dual, hd);
}

TensorHyperDual max(TensorHyperDual& lhs, TensorHyperDual& rhs) {
    auto r = torch::max(lhs.r, rhs.r);
    auto d = torch::where(torch::unsqueeze(lhs.r, -1) > torch::unsqueeze(rhs.r, -1), lhs.d, rhs.d);
    auto hd = torch::where(torch::unsqueeze(lhs.r, -1) > torch::unsqueeze(rhs.r, -1), lhs.hd, rhs.hd);
    return TensorHyperDual(r, d, hd);
}

TensorHyperDual max(torch::Tensor lhs, TensorHyperDual& rhs) {
    auto r = torch::max(lhs, rhs.r);
    auto d = torch::where(lhs > rhs.r, rhs.d, torch::zeros_like(rhs.d));
    auto hd = torch::where(lhs > rhs.r, rhs.hd, torch::zeros_like(rhs.hd));
    return TensorHyperDual(r, d, hd);
}

TensorHyperDual max(TensorHyperDual& lhs, torch::Tensor rhs) {
    auto r = torch::max(lhs.r, rhs);
    auto d = torch::where(lhs.r > rhs, lhs.d, torch::zeros_like(lhs.d));
    auto hd = torch::where(lhs.r > rhs, lhs.hd, torch::zeros_like(lhs.hd));
    return TensorHyperDual(r, d, hd);
}

template<typename Scalar>
TensorHyperDual max(const TensorHyperDual& lhs, Scalar rhs) {
    auto maxres = torch::max(lhs.r, rhs);
    auto dual = torch::where(lhs.r > rhs, lhs.d, torch::zeros_like(lhs.d));
    auto hd = torch::where(lhs.r > rhs, lhs.hd, torch::zeros_like(lhs.hd));
    return TensorHyperDual(std::get<0>(maxres), dual, hd);
}


TensorHyperDual min(TensorHyperDual& lhs, TensorHyperDual& rhs) {
    auto r = torch::min(lhs.r, rhs.r);
    auto d = torch::where(lhs.r < rhs.r, lhs.d, rhs.d);
    auto hd = torch::where(lhs.r < rhs.r, lhs.hd, rhs.hd);
    return TensorHyperDual(r, d, hd);
}

TensorHyperDual min(torch::Tensor lhs, TensorHyperDual& rhs) {
    auto r = torch::min(lhs, rhs.r);
    auto d = torch::where(lhs < rhs.r, rhs.d, torch::zeros_like(rhs.d));
    auto hd = torch::where(lhs < rhs.r, rhs.hd, torch::zeros_like(rhs.hd));
    return TensorHyperDual(r, d, hd);
}


TensorHyperDual min(TensorHyperDual& lhs, torch::Tensor rhs) {
    auto r = torch::min(lhs.r, rhs);
    auto d = torch::where(lhs.r < rhs, lhs.d, torch::zeros_like(lhs.d));
    auto hd = torch::where(lhs.r < rhs, lhs.hd, torch::zeros_like(lhs.hd));
    return TensorHyperDual(r, d, hd);
}

TensorHyperDual sign(TensorHyperDual& td) {
    auto r = torch::sign(td.r);
    auto d = torch::zeros_like(td.d);
    auto hd = torch::zeros_like(td.hd);
    return TensorHyperDual(r, d, hd);
}

template<typename Scalar>
TensorHyperDual min(TensorHyperDual& lhs, Scalar rhs) {
    auto r = torch::min(lhs.r, rhs);
    auto d = torch::where(lhs.r < rhs, lhs.d, torch::zeros_like(lhs.d));
    auto hd = torch::where(lhs.r < rhs, lhs.hd, torch::zeros_like(lhs.hd));
    return TensorHyperDual(r, d, hd);
}




// ... [Other parts of the TensorDual class]

TensorMatHyperDual ger(const TensorHyperDual& x, const TensorHyperDual& y) {

        // Replicate einsum 'mj, mi->mij'
        auto r = torch::einsum("mj, mi->mij", {x.r, y.r});

        // Replicate einsum 'mj, mik->mijk' and 'mjk, mi->mijk'
        //        d1 = torch.einsum('mj, mik->mijk', x.r, y.d)
        auto d1 = torch::einsum("mj, mik->mijk", {x.r, y.d});
        //d2  = torch.einsum('mjk, mi->mijk', x.d, y.r)
        auto d2 = torch::einsum("mjk, mi->mijk", {x.d, y.r});

        // Replicate einsum 'mjk, mik->mijk'
        //        hd1 = torch.einsum('mj, mik->mijk', x.r, y.hd)
        auto hd1 = torch::einsum("mj, mik->mijk", {x.r, y.hd});
        //hd2  = torch.einsum('mjk, mi->mijk', x.d, y.d)
        auto hd2 = torch::einsum("mjk, mi->mijk", {x.d, y.d});

        // Create a TensorMatDual from the results
        return TensorMatHyperDual(r, d1 + d2, hd1 + hd2);
}


}

#endif