#pragma once
#include <torch/torch.h>

#include <type_traits> // For std::is_scalar
#include <vector>
#include <sstream>  // For std::ostringstream
#include <iomanip>   // for std::setprecision
#include <stdexcept>

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



#include "tensordual.hpp"
#include "tensorhyperdual.hpp"
#include "tensormatdual.hpp"
#include "tensormathyperdual.hpp"
namespace janus {


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


// ----------------------------------------------------------------------------
// Return a TensorMatDual whose real part is an N×N identity matrix and whose
// dual part is all-zero.
//
//   r : [N]             (this TensorDual)
//   d : [N,D]
//   ───────────────────────────────────────────────────────────
//   eye()  →  TensorMatDual
//            real : [N,N]            (identity)
//            dual : [N,N,D]          (zeros)
// ----------------------------------------------------------------------------
TensorMatDual TensorDual::eye() const
{   
    /* ----- sanity checks -------------------------------------------------- */
    if (!r.defined() || !d.defined())
        throw std::runtime_error("TensorDual::eye(): tensors are undefined.");
    if (r.dim() != 1 || d.dim() != 2 || r.size(0) != d.size(0))
        throw std::runtime_error("TensorDual::eye(): shapes must be r[N] and d[N,D].");

    const int64_t N = r.size(0);      // matrix size
    const int64_t D = d.size(1);      // dual dimension

    /* ----- real part: identity ------------------------------------------- */
    torch::Tensor r_eye = torch::eye(N, r.options());            // [N,N]

    /* ----- dual part: zeros ---------------------------------------------- */
    torch::Tensor d_zero = torch::zeros({ N, N, D }, d.options()); // [N,N,D]

    return TensorMatDual(r_eye, d_zero);
}


// ─────────────────────────────────────────────────────────────
// Element-wise product   torch::Tensor  *  TensorDual
//
//   • `tensor` may be
//         – scalar          ()          (broadcast to N)
//         – row vector      [N]
//
//   • Result:
//         real : tensor * td.r          [N]
//         dual : tensor * td.d          [N,D]
//
[[nodiscard]]
inline TensorDual operator*(const torch::Tensor& tensor,
                            const TensorDual&    td)
{
    /* ── sanity checks ────────────────────────────────────── */
    if (!tensor.defined())
        throw std::invalid_argument("operator*: lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator*: rhs TensorDual is undefined.");
    if (tensor.device() != td.r.device())
        throw std::invalid_argument("operator*: lhs and rhs must reside on the same device.");

    /* ── broadcastability ---------------------------------- */
    // Accept scalar or 1-D vector length N
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == td.r.size(0))))
        throw std::invalid_argument("operator*: lhs tensor must be scalar or length-N vector.");

    /* ── compute ------------------------------------------- */
    torch::Tensor real_out = tensor * td.r;                   // [N]
    torch::Tensor dual_out = tensor.unsqueeze(-1) * td.d;     // [N,1] * [N,D] → [N,D]

    return TensorDual(std::move(real_out), std::move(dual_out));
}




// ─────────────────────────────────────────────────────────────
// Element-wise quotient   torch::Tensor  /  TensorDual
//
//   • lhs  (tensor) may be scalar (()) or a length-N vector.
//   • rhs  (td)     has r[N] , d[N,D].
//
//   Result
//        real : tensor / td.r                       [N]
//        dual : − tensor / td.r²  ·  td.d           [N,D]
//
[[nodiscard]]
inline TensorDual operator/(const torch::Tensor& tensor,
                            const TensorDual&    td)
{
    /* ── sanity checks ────────────────────────────────────── */
    if (!tensor.defined())
        throw std::invalid_argument("operator/: lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator/: rhs TensorDual is undefined.");
    if (tensor.device() != td.r.device())
        throw std::invalid_argument("operator/: tensors must be on the same device.");

    // scalar or vector of length N
    const int64_t N = td.r.size(0);
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == N)))
        throw std::invalid_argument("operator/: lhs tensor must be scalar or length-N vector.");

    /* ── real part ------------------------------------------------------ */
    torch::Tensor real_out = tensor / td.r;                    // [N]

    /* ── dual part  d = − tensor / r² · d ------------------------------ */
    torch::Tensor coeff = -(tensor / td.r.square())            // [N]
                          .unsqueeze(-1);                      // [N,1] → broadcast on D
    torch::Tensor dual_out = coeff * td.d;                     // [N,1] * [N,D] → [N,D]

    return TensorDual(std::move(real_out), std::move(dual_out));
}

// ─────────────────────────────────────────────────────────────
// Element-wise division:   tensor / TensorHyperDual
//
// Layout after refactor
//   tensor         : ()  or [N]            (scalar or row-vector)
//   td.r           : [N]
//   td.d           : [N,D]
//   td.h           : [N,D,D]
//
// Result
//   real  :  a / g
//   dual  : –a / g²  · g'
//   hyper : –a / g²  · g''  +  2 a / g³ · (g' ⊗ g')
//
inline TensorHyperDual operator/(const torch::Tensor &a,
                                 const TensorHyperDual &td)
{
    /* ── safety checks ────────────────────────────────────── */
    if (!a.defined())
        throw std::invalid_argument("lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined() || !td.h.defined())
        throw std::invalid_argument("rhs TensorHyperDual is undefined.");
    if (a.device() != td.r.device())
        throw std::invalid_argument("tensors must be on the same device.");

    const int64_t N = td.r.size(0);
    const int64_t D = td.d.size(1);

    // lhs must broadcast to length-N vector
    if (!(a.dim() == 0 || (a.dim() == 1 && a.size(0) == N)))
        throw std::invalid_argument("lhs must be scalar or length-N vector.");

    /* ── shorthand ------------------------------------------------------ */
    torch::Tensor g = td.r;   // denom real [N]
    torch::Tensor gp = td.d;  // denom dual [N,D]
    torch::Tensor gpp = td.h; // denom hyper [N,D,D]

    /* ── helpers -------------------------------------------------------- */
    torch::Tensor a_vec = a;               // () or [N]
    torch::Tensor g_inv = 1.0 / g;         // g⁻¹
    torch::Tensor g_inv2 = g_inv * g_inv;  // g⁻²
    torch::Tensor g_inv3 = g_inv2 * g_inv; // g⁻³

    /* ── real part ------------------------------------------------------ */
    torch::Tensor r_out = a_vec * g_inv; // [N]

    /* ── dual part  f' = −a g' / g² --------------------------- */
    torch::Tensor coeff_dual = -(a_vec * g_inv2)    // [N]
                                    .unsqueeze(-1); // [N,1]
    torch::Tensor d_out = coeff_dual * gp;          // [N,D]

    /* ── hyperdual part -------------------------------------- */
    // outer product g'⊗g'  → [N,D,D]
    torch::Tensor gp_outer = gp.unsqueeze(-1) * gp.unsqueeze(-2);

    // term1 = −a g'' / g²
    torch::Tensor term1 = -(a_vec * g_inv2)
                               .unsqueeze(-1)
                               .unsqueeze(-1) *
                          gpp; // [N,D,D]

    // term2 = 2 a (g'⊗g') / g³
    torch::Tensor term2 = 2.0 * (a_vec * g_inv3).unsqueeze(-1).unsqueeze(-1) * gp_outer;

    torch::Tensor h_out = term1 + term2; // [N,D,D]

    return TensorHyperDual(std::move(r_out),
                           std::move(d_out),
                           std::move(h_out));
}

// ─────────────────────────────────────────────────────────────
// Add a tensor to the real part of a TensorDual.
// • lhs may be a scalar (()) or a length-N vector.
// • dual sensitivities remain unchanged.
[[nodiscard]]
inline TensorDual operator+(const torch::Tensor& tensor,
                            const TensorDual&    td)
{
    /* ── sanity checks ────────────────────────────────────── */
    if (!tensor.defined())
        throw std::invalid_argument("operator+: lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator+: rhs TensorDual is undefined.");
    if (tensor.device() != td.r.device())
        throw std::invalid_argument("operator+: tensors must be on the same device.");

    const int64_t N = td.r.size(0);

    // lhs must broadcast to length-N
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == N)))
        throw std::invalid_argument("operator+: lhs tensor must be scalar or length-N vector.");

    /* ── compute real part ────────────────────────────────── */
    torch::Tensor real_out = tensor + td.r;   // broadcast ok

    /* ── dual part unchanged (clone for independence) ────── */
    torch::Tensor dual_out = td.d.clone();    // [N,D]

    return TensorDual(std::move(real_out), std::move(dual_out));
}


// ─────────────────────────────────────────────────────────────
// tensor  +  TensorHyperDual     (tensor on the left)
//
// • tensor may be a scalar (()) or a length-N vector.
// • dual and hyper-dual parts stay identical (cloned for independence).
//
[[nodiscard]]
inline TensorHyperDual operator+(const torch::Tensor& tensor,
                                 const TensorHyperDual& thd)
{
    /* ----- sanity checks ----------------------------------- */
    if (!tensor.defined())
        throw std::invalid_argument("operator+: lhs tensor is undefined.");
    if (!thd.r.defined() || !thd.d.defined() || !thd.h.defined())
        throw std::invalid_argument("operator+: rhs TensorHyperDual is undefined.");
    if (tensor.device() != thd.r.device())
        throw std::invalid_argument("operator+: tensors must be on the same device.");

    const int64_t N = thd.r.size(0);

    // lhs must broadcast to length N
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == N)))
        throw std::invalid_argument("operator+: lhs tensor must be scalar or length-N vector.");

    /* ----- compute real part ---------------------------------------- */
    torch::Tensor r_out = tensor + thd.r;          // broadcast OK

    /* ----- dual & hyper-dual are unchanged (clone for independence) -- */
    torch::Tensor d_out = thd.d.clone();           // [N,D]
    torch::Tensor h_out = thd.h.clone();           // [N,D,D]

    return TensorHyperDual(std::move(r_out),
                           std::move(d_out),
                           std::move(h_out));
}

// ─────────────────────────────────────────────────────────────
// Scalar + TensorDual     (scalar on the left)
//
// Layout after refactor
//     td.r : [N]
//     td.d : [N,D]
//
[[nodiscard]]
inline TensorDual operator+(double scalar, const TensorDual& td)
{
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator+: TensorDual tensors are undefined.");

    /* real part: shift by scalar */
    torch::Tensor r_out = td.r + scalar;   // [N]

    /* dual part: unchanged (clone for independence) */
    torch::Tensor d_out = td.d.clone();    // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}



// ─────────────────────────────────────────────────────────────
// Scalar + TensorHyperDual      (scalar on the left)
//
// Refactored layout
//     r : [N]          d : [N,D]          h : [N,D,D]
//
[[nodiscard]]
inline TensorHyperDual operator+(double scalar, const TensorHyperDual& thd)
{
    if (!thd.r.defined() || !thd.d.defined() || !thd.h.defined())
        throw std::invalid_argument("operator+: TensorHyperDual tensors are undefined.");

    /* real part: shift by scalar */
    torch::Tensor r_out = thd.r + scalar;     // [N]

    /* dual / hyper-dual stay unchanged (clone => independent) */
    torch::Tensor d_out = thd.d.clone();      // [N,D]
    torch::Tensor h_out = thd.h.clone();      // [N,D,D]

    return TensorHyperDual(std::move(r_out),
                           std::move(d_out),
                           std::move(h_out));
}


// ─────────────────────────────────────────────────────────────
// tensor − TensorDual         (tensor on the left)
//
// • tensor may be a scalar (()) or a length‐N vector.
// • TensorDual uses refactored layout   r[N] , d[N,D].
//
//   real_out = tensor − r
//   dual_out = −d
//
[[nodiscard]]
inline TensorDual operator-(const torch::Tensor& tensor,
                            const TensorDual&    td)
{
    /* ── sanity checks ────────────────────────────────────── */
    if (!tensor.defined())
        throw std::invalid_argument("operator-: lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator-: rhs TensorDual is undefined.");
    if (tensor.device() != td.r.device())
        throw std::invalid_argument("operator-: tensors must be on the same device.");

    const int64_t N = td.r.size(0);

    // lhs must broadcast to length-N
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == N)))
        throw std::invalid_argument("operator-: lhs tensor must be scalar or length-N vector.");

    /* ── compute real part --------------------------------- */
    torch::Tensor real_out = tensor - td.r;      // broadcast OK   [N]

    /* ── dual part: negate (clone for independence) -------- */
    torch::Tensor dual_out = -td.d.clone();      // [N,D]

    return TensorDual(std::move(real_out), std::move(dual_out));
}




// ─────────────────────────────────────────────────────────────
// tensor − TensorHyperDual        (tensor on the left)
//
// Refactored shapes
//     td.r : [N]
//     td.d : [N, D]
//     td.h : [N, D, D]
//
// The dual and hyper-dual parts are simply negated (no new
// sensitivities from the plain tensor).
//
[[nodiscard]]
inline TensorHyperDual operator-(const torch::Tensor& tensor,
                                 const TensorHyperDual& td)
{
    /* -------- sanity checks ------------------------------------------ */
    if (!tensor.defined())
        throw std::invalid_argument("operator-: lhs tensor is undefined.");
    if (!td.r.defined() || !td.d.defined() || !td.h.defined())
        throw std::invalid_argument("operator-: rhs TensorHyperDual is undefined.");
    if (tensor.device() != td.r.device())
        throw std::invalid_argument("operator-: tensors must be on the same device.");

    const int64_t N = td.r.size(0);

    // lhs must broadcast to length-N
    if (!(tensor.dim() == 0 || (tensor.dim() == 1 && tensor.size(0) == N)))
        throw std::invalid_argument("operator-: lhs tensor must be scalar or length-N vector.");

    /* -------- real part ---------------------------------------------- */
    torch::Tensor r_out = tensor - td.r;      // scalar/vec − [N]  → [N]

    /* -------- dual & hyper-dual: negate and clone -------------------- */
    torch::Tensor d_out = -td.d.clone();      // [N,D]
    torch::Tensor h_out = -td.h.clone();      // [N,D,D]

    return TensorHyperDual(std::move(r_out),
                           std::move(d_out),
                           std::move(h_out));
}


// ─────────────────────────────────────────────────────────────
// Left  multiply (row-vector)  :  td * M   where  td.r[P], M[P,Q]  → size Q
// Right multiply (column-vec)  :  td * M   where  td.r[Q], M[P,Q]  → size P
//
[[nodiscard]]
inline TensorDual operator*(const TensorDual& td,
                            const TensorMatDual& M)
{
    const auto P = td.r.size(0);      // length of the vector
    const auto D = td.d.size(1);      // dual dimension

    torch::Tensor r_out, d_out;

    /* ---------- left multiplication  (length matches rows) ------------ */
    if (P == M.r.size(0)) {           // M real shape [P,Q]

        const auto Q = M.r.size(1);

        // real :   r_j = Σ_i  td_i · M_{i j}
        r_out = torch::einsum("i,ij->j", { td.r, M.r });          // [Q]

        // dual :   d_{j d} = Σ_i  td_i · M_{i j d} + td_{i d} · M_{i j}
        torch::Tensor term1 = torch::einsum("i,ijd->jd", { td.r, M.d });  // [Q,D]
        torch::Tensor term2 = torch::einsum("id,ij->jd", { td.d, M.r });  // [Q,D]
        d_out = term1 + term2;                                           // [Q,D]
    }
    /* ---------- right multiplication (length matches cols) ------------ */
    else if (P == M.r.size(1)) {      // M real shape [Q,P]

        const auto Q = M.r.size(0);

        // real :   r_j = Σ_i  M_{j i} · td_i
        r_out = torch::einsum("ji,i->j", { M.r, td.r });          // [Q]

        // dual :   d_{j d} = Σ_i  M_{j i d} · td_i + M_{j i} · td_{i d}
        torch::Tensor term1 = torch::einsum("jid,i->jd", { M.d, td.r });  // [Q,D]
        torch::Tensor term2 = torch::einsum("ji,id->jd", { M.r, td.d });  // [Q,D]
        d_out = term1 + term2;                                           // [Q,D]
    }
    else {
        throw std::invalid_argument(
            "TensorDual × TensorMatDual: incompatible dimensions.");
    }

    return TensorDual(std::move(r_out), std::move(d_out));
}

// ─────────────────────────────────────────────────────────────
// Multiply a sensitivity-aware matrix by a sensitivity-aware
// vector.  Batch-free layout:
//
//   Matrix (tmd):
//        r : [P,Q]
//        d : [P,Q,D]
//
//   Vector (v = other):
//        r : [L]       (length L)
//        d : [L,D]
//
// Supported cases:
//
//   1.  P×Q  ·  Q      → length P      (standard right multiplication)
//   2.  P×Q  ·  P      → length Q      (treat vector as row, i.e. vᵀ·M)
//
// In both cases the product rule is:
//
//     rᵢ   = Σ_j   Mᵢⱼ  vⱼ
//     dᵢd  = Σ_j ( Mᵢⱼ  v'ⱼd   +   M'ᵢⱼd  vⱼ )
//
[[nodiscard]]
inline TensorDual operator*(const TensorMatDual& M,
                            const TensorDual&   v)
{
    torch::Tensor r_out, d_out;

    const auto P = M.r.size(0);          // rows
    const auto Q = M.r.size(1);          // cols
    const auto L = v.r.size(0);          // vector length
    const auto D = v.d.size(1);          // dual dim

    /* ----- Case 1:  right multiplication  M[P,Q] · v[Q] ----- */
    if (L == Q) {
        // real part
        r_out = torch::einsum("ij,j->i",  { M.r,  v.r });          // [P]

        // dual part
        auto term_Md = torch::einsum("ijd,j->id", { M.d, v.r });   // M'·v
        auto term_vd = torch::einsum("ij,jd->id", { M.r, v.d });   // M ·v'
        d_out = term_Md + term_vd;                                 // [P,D]
    }
    /* ----- Case 2:  left  multiplication  M[P,Q] · v[P] ----- */
    else if (L == P) {
        // real part  (v treated as row-vector on the left)
        r_out = torch::einsum("ij,i->j",  { M.r,  v.r });          // [Q]

        // dual part
        auto term_Md = torch::einsum("ijd,i->jd", { M.d, v.r });   // M'·v
        auto term_vd = torch::einsum("ij,id->jd", { M.r, v.d });   // M ·v'
        d_out = term_Md + term_vd;                                 // [Q,D]
    }
    else {
        throw std::invalid_argument(
            "TensorMatDual * TensorDual: incompatible dimensions.");
    }

    return TensorDual(std::move(r_out), std::move(d_out));
}

// ─────────────────────────────────────────────────────────────
// Element-wise product of two TensorMatDuals (batch-free)
//
//   r  : [N,L]
//   d  : [N,L,D]
//
// Result uses the product rule:
//     rₒ = r₁ ⊙ r₂
//     dₒ = r₁ ⊙ d₂  +  d₁ ⊙ r₂
//
[[nodiscard]]
inline TensorMatDual operator*(const TensorMatDual& A,
                               const TensorMatDual& B)
{
    /* ── shape check ──────────────────────────────────────── */
    if (A.r.sizes() != B.r.sizes())
        throw std::invalid_argument("TensorMatDual * TensorMatDual: real parts have incompatible shapes.");

    /* ── real part: element-wise product -------------------- */
    torch::Tensor r_out = A.r * B.r;                           // [N,L]

    /* ── dual part: product rule ---------------------------- */
    // broadcast scalars over the dual axis
    torch::Tensor d_out = A.r.unsqueeze(-1) * B.d              // r₁ ⊙ d₂
                        + A.d * B.r.unsqueeze(-1);             // d₁ ⊙ r₂
    // shape [N,L,D]

    return TensorMatDual(std::move(r_out), std::move(d_out));
}



// ─────────────────────────────────────────────────────────────
// Scalar − TensorDual      (scalar on the left)
//
//   td.r : [N]      td.d : [N,D]
//
//   real_out = scalar − td.r
//   dual_out = −td.d
//
[[nodiscard]]
inline TensorDual operator-(int scalar, const TensorDual& td)
{
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator-: TensorDual tensors are undefined.");

    /* real part: broadcast scalar over the vector */
    torch::Tensor r_out = scalar - td.r;        // [N]

    /* dual part: negate (clone for independence if needed) */
    torch::Tensor d_out = -td.d.clone();        // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}

// ─────────────────────────────────────────────────────────────
// Scalar ÷ TensorDual        (scalar on the left)
//
// Layout (batch-free)
//     td.r : [N]
//     td.d : [N,D]
//
// Chain-rule for f(x) = c / x:
//     f' = −c / x²  ·  x'
//
[[nodiscard]]
inline TensorDual operator/(double scalar, const TensorDual& td)
{
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator/: TensorDual tensors are undefined.");

    /* ----- real part --------------------------------------------------- */
    torch::Tensor r_out = scalar / td.r;                     // [N]

    /* ----- dual part  d = −c / r² · d_r ------------------------------- */
    torch::Tensor coeff = (-scalar / td.r.square())          // [N]
                          .unsqueeze(-1);                    // [N,1] → broadcast on D
    torch::Tensor d_out = coeff * td.d;                      // [N,1] * [N,D] → [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}


// r : [N]      d : [N,D]
[[nodiscard]]
inline TensorDual operator*(double scalar, const TensorDual& td)
{
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("operator*: TensorDual tensors are undefined.");

    torch::Tensor real_out = td.r * scalar;   // scale vector
    torch::Tensor dual_out = td.d * scalar;   // scale sensitivities

    return TensorDual(std::move(real_out), std::move(dual_out));
}

// ─────────────────────────────────────────────────────────────
// Scalar × TensorMatDual  (scalar on the left)
//
// Scales both the real matrix and its sensitivity stack.
[[nodiscard]]
inline TensorMatDual operator*(double scalar, const TensorMatDual& tmd)
{
    if (!tmd.r.defined() || !tmd.d.defined())
        throw std::invalid_argument("operator*: TensorMatDual tensors are undefined.");

    torch::Tensor r_out = tmd.r * scalar;   // [N,L]   element-wise
    torch::Tensor d_out = tmd.d * scalar;   // [N,L,D] element-wise

    return TensorMatDual(std::move(r_out), std::move(d_out));
}


// ─────────────────────────────────────────────────────────────
// Element-wise power for batch-free TensorDuals
//
//   base:      r = a[N]       d = b[N,D]
//   exponent:  r = c[N]       d = d[N,D]
//
//   x = a + b ε ,  y = c + d ε
//   x^y = a^c + [ a^c·d·ln a  +  a^(c-1)·b·c ] ε
//
[[nodiscard]]
inline TensorDual pow(const TensorDual& base,
                      const TensorDual& exponent)
{
    if (!base.r.defined() || !base.d.defined() ||
        !exponent.r.defined() || !exponent.d.defined())
        throw std::invalid_argument("pow(TensorDual,TensorDual): operands undefined.");

    const auto& a = base.r;          // [N]
    const auto& b = base.d;          // [N,D]
    const auto& c = exponent.r;      // [N]
    const auto& d = exponent.d;      // [N,D]

    /* ----- real part --------------------------------------------------- */
    torch::Tensor real = torch::pow(a, c);                    // a^c   [N]

    /* ----- dual part --------------------------------------------------- */
    // term1 = a^c · ln(a) · d
    torch::Tensor term1 = (real * torch::log(a)).unsqueeze(-1) * d;     // [N,1]*[N,D] → [N,D]

    // term2 = a^(c-1) · b · c   ==   a^c · (c/a) · b
    torch::Tensor term2 = (real * (c / a)).unsqueeze(-1) * b;           // [N,D]

    torch::Tensor dual = term1 + term2;                                 // [N,D]

    return TensorDual(std::move(real), std::move(dual));
}


// ─────────────────────────────────────────────────────────────
// Element-wise maximum of two TensorDuals
//
// Layout
//     r : [N]          d : [N,D]
//
// At each position i
//     if lhs.r[i] >= rhs.r[i]   → take lhs.d[i,⋅]
//     else                      → take rhs.d[i,⋅]
//
[[nodiscard]]
inline TensorDual max(const TensorDual& lhs,
                      const TensorDual& rhs)
{
    /* ----- sanity checks ----------------------------------- */
    if (!lhs.r.defined() || !lhs.d.defined() ||
        !rhs.r.defined() || !rhs.d.defined())
        throw std::invalid_argument("max(TensorDual,TensorDual): inputs undefined.");

    if (lhs.r.sizes() != rhs.r.sizes() ||
        lhs.d.sizes() != rhs.d.sizes())
        throw std::invalid_argument("max(TensorDual,TensorDual): shape mismatch.");

    /* ----- real part: element-wise maximum ----------------- */
    torch::Tensor r_out = torch::max(lhs.r, rhs.r);        // [N]

    /* ----- dual part: choose by mask ----------------------- */
    torch::Tensor mask = (lhs.r >= rhs.r)                  // [N] bool
                         .unsqueeze(-1);                   // [N,1]  → broadcast on D

    torch::Tensor d_out = torch::where(mask, lhs.d, rhs.d);  // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}

// ─────────────────────────────────────────────────────────────
// Element-wise max between a TensorDual and a plain tensor.
//
// Layout (batch-free)
//     lhs.r : [N]        lhs.d : [N,D]
//     rhs   : ()  or [N]         (scalar or row vector)
//
// Result
//     r = max(lhs.r , rhs)
//     d = lhs.d  where lhs.r >= rhs,  else 0
//
[[nodiscard]]
inline TensorDual max(const TensorDual& lhs,
                      const torch::Tensor& rhs)
{
    /* ── checks ───────────────────────────────────────────── */
    if (!lhs.r.defined() || !lhs.d.defined())
        throw std::invalid_argument("max(TensorDual,tensor): TensorDual undefined.");
    if (!rhs.defined())
        throw std::invalid_argument("max(TensorDual,tensor): rhs tensor undefined.");
    if (rhs.device() != lhs.r.device())
        throw std::invalid_argument("max(TensorDual,tensor): tensors on different devices.");

    const int64_t N = lhs.r.size(0);

    // rhs must broadcast to length N
    if (!(rhs.dim() == 0 || (rhs.dim() == 1 && rhs.size(0) == N)))
        throw std::invalid_argument("max(TensorDual,tensor): rhs must be scalar or length-N vector.");

    /* ── broadcast rhs if it is scalar ───────────────────── */
    torch::Tensor rhs_vec = (rhs.dim() == 0) ? rhs.expand({N}) : rhs;   // [N]

    /* ── real part  r = max(a , b) ───────────────────────── */
    torch::Tensor r_out = torch::max(lhs.r, rhs_vec);                   // [N]

    /* ── mask where lhs wins ─────────────────────────────── */
    torch::Tensor mask = (lhs.r >= rhs_vec).unsqueeze(-1);              // [N,1]

    /* ── dual part: keep lhs.d where mask, else zero ─────── */
    torch::Tensor d_out = torch::where(mask, lhs.d, torch::zeros_like(lhs.d)); // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}



// ─────────────────────────────────────────────────────────────
// Element-wise minimum of two TensorDuals
//
// Layout (batch-free)
//   r : [N]          d : [N,D]
//
// Result
//   rᵢ = min(lhs.rᵢ , rhs.rᵢ)
//   dᵢ = lhs.dᵢ   if lhs.rᵢ ≤ rhs.rᵢ
//        rhs.dᵢ   otherwise
//
[[nodiscard]]
inline TensorDual min(const TensorDual& lhs,
                      const TensorDual& rhs)
{
    /* ----- checks ----------------------------------------------------- */
    if (!lhs.r.defined() || !lhs.d.defined() ||
        !rhs.r.defined() || !rhs.d.defined())
        throw std::invalid_argument("min(TensorDual,TensorDual): inputs undefined.");

    if (lhs.r.sizes() != rhs.r.sizes() ||
        lhs.d.sizes() != rhs.d.sizes())
        throw std::invalid_argument("min(TensorDual,TensorDual): shape mismatch.");

    /* ----- real part -------------------------------------------------- */
    torch::Tensor r_out = torch::min(lhs.r, rhs.r);            // [N]

    /* ----- mask & dual selection ------------------------------------- */
    torch::Tensor mask = (lhs.r <= rhs.r).unsqueeze(-1);        // [N,1]
    torch::Tensor d_out = torch::where(mask, lhs.d, rhs.d);     // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}




// ─────────────────────────────────────────────────────────────
// Element-wise minimum between a TensorDual and a plain tensor.
//
// Layout
//     lhs.r : [N]          lhs.d : [N,D]
//     rhs   : ()  or [N]   (scalar or length-N vector)
//
// Result
//     rᵢ = min(lhs.rᵢ , rhsᵢ)
//     dᵢ = lhs.dᵢ   if lhs.rᵢ ≤ rhsᵢ
//          0        otherwise
//
[[nodiscard]]
inline TensorDual min(const TensorDual& lhs,
                      const torch::Tensor& rhs)
{
    /* ── sanity checks ────────────────────────────────────── */
    if (!lhs.r.defined() || !lhs.d.defined())
        throw std::invalid_argument("min(TensorDual,tensor): TensorDual undefined.");
    if (!rhs.defined())
        throw std::invalid_argument("min(TensorDual,tensor): rhs tensor undefined.");
    if (rhs.device() != lhs.r.device())
        throw std::invalid_argument("min(TensorDual,tensor): tensors on different devices.");

    const int64_t N = lhs.r.size(0);

    // rhs must broadcast to length N
    if (!(rhs.dim() == 0 || (rhs.dim() == 1 && rhs.size(0) == N)))
        throw std::invalid_argument("min(TensorDual,tensor): rhs must be scalar or length-N vector.");

    /* ── broadcast rhs if scalar ─────────────────────────── */
    torch::Tensor rhs_vec = (rhs.dim() == 0) ? rhs.expand({N}) : rhs;   // [N]

    /* ── real part  r = min(lhs.r , rhs) ─────────────────── */
    torch::Tensor r_out = torch::min(lhs.r, rhs_vec);                  // [N]

    /* ── mask where lhs wins ─────────────────────────────── */
    torch::Tensor mask = (lhs.r <= rhs_vec).unsqueeze(-1);             // [N,1] bool

    /* ── dual part: keep lhs.d where mask, else 0 ────────── */
    torch::Tensor d_out = torch::where(mask, lhs.d, torch::zeros_like(lhs.d)); // [N,D]

    return TensorDual(std::move(r_out), std::move(d_out));
}



/**
 * Element-wise sign for a TensorDual.
 *
 *   real_out  =  sgn(r)          ∈ {-1, 0, +1}
 *   dual_out  =  0               (∂ sgn/∂x is 0 almost everywhere;
 *                                 we choose 0 even when r == 0)
 */
[[nodiscard]]
inline TensorDual sign(const TensorDual& td)
{
    if (!td.r.defined() || !td.d.defined())
        throw std::invalid_argument("sign(TensorDual): tensors undefined.");

    torch::Tensor r_out = torch::sgn(td.r);         // [N]
    torch::Tensor d_out = torch::zeros_like(td.d);  // [N,D]  derivative vanishes

    return TensorDual(std::move(r_out), std::move(d_out));
}


/**
 * Element-wise exponentiation of a TensorDual by a plain tensor.
 *
 *   base.r : [N]      base.d : [N,D]
 *   exponent : () or [N]
 *
 * For  f(x) = x^c  with  x = a + b ε  (c has no dual part)
 *     f  = a^c
 *     f' = c · a^{c-1} · b
 */
[[nodiscard]]
inline TensorDual pow(const TensorDual& base,
                      const torch::Tensor& exponent)
{
    /* ---------- checks ------------------------------------------------ */
    if (!base.r.defined() || !base.d.defined())
        throw std::invalid_argument("pow(TensorDual,tensor): TensorDual undefined.");
    if (!exponent.defined())
        throw std::invalid_argument("pow(TensorDual,tensor): exponent tensor undefined.");
    if (exponent.device() != base.r.device())
        throw std::invalid_argument("pow(TensorDual,tensor): tensors on different devices.");

    const int64_t N = base.r.size(0);

    // exponent must broadcast to length N
    if (!(exponent.dim() == 0 || (exponent.dim() == 1 && exponent.size(0) == N)))
        throw std::invalid_argument("pow(TensorDual,tensor): exponent must be scalar or length-N vector.");

    /* ---------- real part -------------------------------------------- */
    torch::Tensor real = torch::pow(base.r, exponent);                // [N]

    /* ---------- dual part: c · a^{c-1} · b --------------------------- */
    torch::Tensor factor = exponent * torch::pow(base.r, exponent - 1); // [N]
    torch::Tensor dual  = factor.unsqueeze(-1) * base.d;                // [N,1] * [N,D] → [N,D]

    return TensorDual(std::move(real), std::move(dual));
}

/**
 * Raise each element of a TensorDual to a **scalar** power.
 *
 *   base.r : [N]
 *   base.d : [N,D]
 *   exponent : scalar  c
 *
 * For  f(x) = x^c  with  x = a + b ε
 *      f  = a^c
 *      f' = c · a^{c-1} · b
 */
[[nodiscard]]
inline TensorDual pow(const TensorDual& base, double exponent)
{
    if (!base.r.defined() || !base.d.defined())
        throw std::invalid_argument("pow(TensorDual,double): base tensors undefined.");

    /* -------- real part --------------------------------------------- */
    torch::Tensor real = torch::pow(base.r, exponent);          // [N]

    /* -------- dual part  f' = c·a^{c-1}·b --------------------------- */
    torch::Tensor coeff = exponent * torch::pow(base.r, exponent - 1); // [N]
    torch::Tensor dual  = coeff.unsqueeze(-1) * base.d;                // [N,1]*[N,D] → [N,D]

    return TensorDual(std::move(real), std::move(dual));
}



/**
 * Generalised outer-product (`ger`) of two TensorDuals, batch-free.
 *
 *  x.r : [P]      x.d : [P,D]
 *  y.r : [Q]      y.d : [Q,D]
 *
 * Result (TensorMatDual)
 *     r[P,Q]           =  x.r ⊗ y.r
 *     d[P,Q,D]         =  x.r ⊗ y.d  +  x.d ⊗ y.r
 */
[[nodiscard]]
inline TensorMatDual ger(const TensorDual& x,
                         const TensorDual& y)
{
    /* -------- sanity checks ----------------------------------------- */
    if (!x.r.defined() || !x.d.defined() ||
        !y.r.defined() || !y.d.defined())
        throw std::invalid_argument(
            "ger(TensorDual,TensorDual): operands undefined.");

    const int64_t P = x.r.size(0);          // |x|
    const int64_t Q = y.r.size(0);          // |y|
    const int64_t D = x.d.size(1);

    if (y.d.size(1) != D)
        throw std::invalid_argument("ger: dual dimensions do not match.");

    /* -------- real part  r = x.r ⊗ y.r ------------------------------ */
    torch::Tensor r = x.r.unsqueeze(1) * y.r.unsqueeze(0);            // [P,Q]

    /* -------- dual part -------------------------------------------- */
    // term1 = x.r ⊗ y.d           → [P,1,1] · [1,Q,D] = [P,Q,D]
    torch::Tensor term1 = x.r.unsqueeze(1).unsqueeze(2)               // [P,1,1]
                         * y.d.unsqueeze(0);                          // [1,Q,D]

    // term2 = x.d ⊗ y.r           → [P,1,D] · [1,Q,1] = [P,Q,D]
    torch::Tensor term2 = x.d.unsqueeze(1)                            // [P,1,D]
                         * y.r.unsqueeze(0).unsqueeze(2);             // [1,Q,1]

    torch::Tensor d = term1 + term2;                                  // [P,Q,D]

    return TensorMatDual(std::move(r), std::move(d));
}

}