#pragma once

#include <torch/torch.h>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include "tensordual.hpp"
#include "tensorhyperdual.hpp" // for TensorHyperDual
namespace janus {
class TensorMatHyperDual : public torch::CustomClassHolder {
    public:
        // ── data ────────────────────────────────────────────────
        torch::Tensor r;   ///< real       [N, L]
        torch::Tensor d;   ///< dual       [N, L, D]
        torch::Tensor h;   ///< hyperdual  [N, L, H1, H2]
        torch::Dtype  dtype_  = torch::kFloat64;
        torch::Device device_ = torch::kCPU;
    
        // ── constructor ────────────────────────────────────────
        TensorMatHyperDual(torch::Tensor r_,
                           torch::Tensor d_,
                           torch::Tensor h_)
            : r(std::move(r_)), d(std::move(d_)), h(std::move(h_)),
              dtype_(torch::typeMetaToScalarType(r.dtype())),
              device_(r.device())
        {
            // shape checks
            if (r.dim() != 2)
                throw std::invalid_argument("r must be 2-D [N,L].");
            if (d.dim() != 3)
                throw std::invalid_argument("d must be 3-D [N,L,D].");
            if (h.dim() != 4)
                throw std::invalid_argument("h must be 4-D [N,L,H1,H2].");
    
            if (r.size(0) != d.size(0) || r.size(0) != h.size(0) ||
                r.size(1) != d.size(1) || r.size(1) != h.size(1))
                throw std::invalid_argument("First two dims (N,L) must match across r, d, h.");
        }

        // perfect-forwarding ctor: works for lvalue *and* rvalue tensors
        template<typename TR, typename TD, typename TH>
        TensorMatHyperDual(TR&& real, TD&& dual, TH&& hyper)
            : r(std::forward<TR>(real)), d(std::forward<TD>(dual)), h(std::forward<TH>(hyper)) {}

        // ── convenience accessors ───────────────────────────────
        inline int64_t rows() const { return r.size(0); }      // N
        inline int64_t cols() const { return r.size(1); }      // L
        inline int64_t dual_dim()   const { return d.size(2); }      // D
        inline int64_t h1_dim()     const { return h.size(2); }      // H1  (shared with L)
        inline int64_t h2_dim()     const { return h.size(3); }      // H2
    
        // ─────────────────────────────────────────────────────────────
        // Convert a tensor’s shape to a human-readable string, e.g. "[3, 5, 7]"
        [[nodiscard]] inline static std::string sizes_to_string(const torch::Tensor& t) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < t.dim(); ++i) {
                oss << t.size(i);
                if (i + 1 < t.dim()) oss << ", ";
            }
            oss << "]";
            return oss.str();
        }
        
        // ─────────────────────────────────────────────────────────────
        // Human-readable summary of the tensor shapes.
        [[nodiscard]] inline std::string toString() const {
            std::ostringstream oss;
            oss << "TensorMatHyperDual("
                << "r: " << sizes_to_string(r) << ", "
                << "d: " << sizes_to_string(d) << ", "
                << "h: " << sizes_to_string(h) << ")";
            return oss.str();
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Return a copy of this TensorMatHyperDual on a different device.
        [[nodiscard]] inline TensorMatHyperDual to(const torch::Device& device) const {
            // Fast path: already on the requested device.
            if (device == device_) return *this;
    
            return TensorMatHyperDual(r.to(device),   // real      [N,L]
                                    d.to(device),   // dual      [N,L,D]
                                    h.to(device));  // hyperdual [N,L,H1,H2]
        }
    
        // ─────────────────────────────────────────────────────────────
        // Query the storage device (CPU / CUDA, etc.)
        [[nodiscard]] inline torch::Device device() const noexcept {
            return device_;
        }
    
        // Promote a TensorDual (first-order sensitivities of a vector) to a
        // TensorMatHyperDual with one singleton matrix axis.
        explicit TensorMatHyperDual(const TensorDual& x, int dim = 1)
            : dtype_(torch::typeMetaToScalarType(x.r.dtype())),
            device_(x.r.device())
        {
            if (!x.r.defined() || !x.d.defined())
                throw std::invalid_argument("TensorDual tensors are undefined.");
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("dim must be 0 (row) or 1 (col).");
    
            const auto D = x.d.size( (x.d.dim() == 2) ? 1 : 2 );   // sensitivity count
    
            if (dim == 0) {                  // make a row-vector
                r = x.r.unsqueeze(0);                       // [1,L]
                d = x.d.unsqueeze(0);                       // [1,L,D]
                h = torch::zeros({1, r.size(1), D, D}, x.r.options()); // [1,L,D,D]
            } else {                       // dim == 1 → column-vector
                r = x.r.unsqueeze(1);                       // [N,1]
                d = x.d.unsqueeze(1);                       // [N,1,D]
                h = torch::zeros({r.size(0), 1, D, D}, x.r.options()); // [N,1,D,D]
            }
        }
    
        
        // ─────────────────────────────────────────────────────────────
        // Promote to complex dtype (if not already)
        [[nodiscard]] inline TensorMatHyperDual complex() const {
            auto promote = [](const torch::Tensor& t) {
                return t.is_complex() ? t
                                    : torch::complex(t, torch::zeros_like(t));
            };
    
            return TensorMatHyperDual(promote(r), promote(d), promote(h));
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Strip any imaginary part, keeping shapes unchanged.
        [[nodiscard]] inline TensorMatHyperDual real() const {
            auto to_real = [](const torch::Tensor& t) {
                return t.is_complex() ? torch::real(t) : t;
            };
    
            return TensorMatHyperDual(to_real(r), to_real(d), to_real(h));
        }
    
        // ─────────────────────────────────────────────────────────────
        // Return the pure-imaginary components (zero if tensors are real).
        [[nodiscard]] inline TensorMatHyperDual imag() const {
            auto to_imag = [](const torch::Tensor& t) {
                return t.is_complex() ? torch::imag(t) : torch::zeros_like(t);
            };
    
            return TensorMatHyperDual(to_imag(r), to_imag(d), to_imag(h));
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise absolute value with first- and second-order propagation.
        //
        //   r_out = |r|
        //   d_out = sgn(r) · d
        //   h_out = 0                (|·| is not twice-differentiable at 0)
        [[nodiscard]] inline TensorMatHyperDual abs() const {
            torch::Tensor r_abs  = torch::abs(r);          // [N,L]
    
            torch::Tensor sign   = torch::sgn(r);          // [N,L]
            torch::Tensor d_abs  = sign.unsqueeze(-1) * d; // broadcast on D  → [N,L,D]
    
            torch::Tensor h_zero = torch::zeros_like(h);   // [N,L,D,D]
    
            return TensorMatHyperDual(r_abs, d_abs, h_zero);
        }
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Row-/column-wise max with first- and second-order propagation.
        //
        //   • dim = 0  → max over rows  → r:[1,L] , d:[1,L,D] , h:[1,L,D,D]
        //   • dim = 1  → max over cols  → r:[N,1] , d:[N,1,D] , h:[N,1,D,D]
        //
        [[nodiscard]] inline TensorMatHyperDual max(int dim = 1) const {
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("TensorMatHyperDual::max(): dim must be 0 or 1.");
    
            // ── max on real part ────────────────────────────────────
            auto max_pair = torch::max(r, /*dim=*/dim, /*keepdim=*/true);
            torch::Tensor r_max = std::get<0>(max_pair);          // [1,L] or [N,1]
            torch::Tensor idx   = std::get<1>(max_pair);          // same shape, long
    
            // ── gather matching dual entries ───────────────────────
            auto idx_d = idx.unsqueeze(-1)            // [*,*,1]
                                .expand_as(d.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
                                // expands to [*,*,D]
            torch::Tensor d_max = torch::gather(d, dim, idx_d);   // [1,L,D] or [N,1,D]
    
            // ── gather matching hyperdual entries ──────────────────
            auto idx_h = idx.unsqueeze(-1)             // [*,*,1]
                        .unsqueeze(-1)               // [*,*,1,1]
                        .expand_as(h.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                            0, torch::indexing::Slice(), torch::indexing::Slice()}));
                        // → [*,*,D,D]
            torch::Tensor h_max = torch::gather(h, dim, idx_h);   // [1,L,D,D] or [N,1,D,D]
    
            return TensorMatHyperDual(r_max, d_max, h_max);
        }
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Row- or column-wise minimum with sensitivity propagation
        //
        //   dim = 0  → min over rows  → r:[1,L] , d:[1,L,D] , h:[1,L,D,D]
        //   dim = 1  → min over cols  → r:[N,1] , d:[N,1,D] , h:[N,1,D,D]
        //
        [[nodiscard]] TensorMatHyperDual min(int dim = 1) const {
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("TensorMatHyperDual::min(): dim must be 0 or 1.");
    
            /* 1 — real part and arg-min indices */
            auto pair   = torch::min(r, dim, /*keepdim=*/true);
            torch::Tensor r_min = std::get<0>(pair);          // [1,L] or [N,1]
            torch::Tensor idx   = std::get<1>(pair);          // same shape (Long)
    
            /* 2 — gather matching dual slice */
            auto idx_d = idx.unsqueeze(-1)                    // [*,*,1]
                            .expand({-1, -1, d.size(2)});  // → [*,*,D]
            torch::Tensor d_min = torch::gather(d, dim, idx_d);   // [1,L,D] / [N,1,D]
    
            /* 3 — gather matching hyperdual slice */
            auto idx_h = idx.unsqueeze(-1)                    // [*,*,1]
                            .unsqueeze(-1)                   // [*,*,1,1]
                            .expand({-1, -1, h.size(2), h.size(3)}); // [*,*,D,D]
            torch::Tensor h_min = torch::gather(h, dim, idx_h);       // [1,L,D,D] / [N,1,D,D]
    
            return TensorMatHyperDual(r_min, d_min, h_min);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Sum along a matrix axis (rows or columns), keepdim = true.
        //
        //   • dim = 0  → r:[1,L] , d:[1,L,D] , h:[1,L,D,D]
        //   • dim = 1  → r:[N,1] , d:[N,1,D] , h:[N,1,D,D]
        //
        [[nodiscard]] inline TensorMatHyperDual sum(int dim) const {
            // allow negative dims (Python-style) → convert to positive
            if (dim < 0) dim += r.dim();                // r.dim() == 2
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("TensorMatHyperDual::sum(): dim must be 0 or 1.");
    
            torch::Tensor r_sum = r.sum(dim, /*keepdim=*/true);  // [1,L] or [N,1]
            torch::Tensor d_sum = d.sum(dim, /*keepdim=*/true);  // [1,L,D] / [N,1,D]
            torch::Tensor h_sum = h.sum(dim, /*keepdim=*/true);  // [1,L,D,D] / [N,1,D,D]
    
            return TensorMatHyperDual(r_sum, d_sum, h_sum);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise square
        //   r_out = r²
        //   d_out = 2 r · d
        //   h_out = 2 (d ⊗ d  +  r · h)
        [[nodiscard]] inline TensorMatHyperDual square() const {
            /* ── real part ───────────────────────────────────────── */
            torch::Tensor r_sq = r.square();                       // [N,L]
    
            /* ── dual part : 2 r d ──────────────────────────────── */
            torch::Tensor d_sq = 2.0 * r.unsqueeze(-1) * d;        // [N,L,1] * [N,L,D] → [N,L,D]
    
            /* ── hyperdual part : 2( d⊗d + r h ) ────────────────── */
            // d⊗d  → outer product along the D axis: [N,L,D,1] * [N,L,1,D] = [N,L,D,D]
            torch::Tensor dd_outer = d.unsqueeze(-1) * d.unsqueeze(-2);
    
            // r·h  → broadcast r across the two sensitivity axes
            torch::Tensor rh_term  = r.unsqueeze(-1).unsqueeze(-1) * h;   // [N,L,1,1] * [N,L,D,D]
    
            torch::Tensor h_sq = 2.0 * (dd_outer + rh_term);       // [N,L,D,D]
    
            return TensorMatHyperDual(r_sq, d_sq, h_sq);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Pretty-print a TensorMatHyperDual
        friend std::ostream &operator<<(std::ostream &os,
                                        const TensorMatHyperDual &obj)
        {
            os << "TensorMatHyperDual {\n"
               << "  dtype : " << torch::toString(obj.dtype_) << '\n'
               << "  device: " << obj.device_ << '\n'
               << "  r     : " << sizes_to_string(obj.r) << '\n'
               << "  d     : " << sizes_to_string(obj.d) << '\n'
               << "  h     : " << sizes_to_string(obj.h) << '\n'
               << "}";
            return os;
        }
        
        // Forward declaration for eye function
        TensorMatHyperDual eye();
    
        // ─────────────────────────────────────────────────────────────
        // Remove all singleton axes while preserving required ranks.
        //
        //   • r must end up 2-D  [N,L]
        //   • d must end up 3-D  [N,L,D]
        //   • h must end up 4-D  [N,L,D,D]
        //
        [[nodiscard]] TensorMatHyperDual squeeze() const {
            torch::Tensor r_sq = r.squeeze();     // rank ≤ 2
            torch::Tensor d_sq = d.squeeze();     // rank ≤ 3
            torch::Tensor h_sq = h.squeeze();     // rank ≤ 4
    
            /* ── ensure r is 2-D ─────────────────────────────────── */
            while (r_sq.dim() < 2) r_sq = r_sq.unsqueeze(-1);  // add trailing 1’s
    
            /* ── ensure d is 3-D ─────────────────────────────────── */
            while (d_sq.dim() < 3) d_sq = d_sq.unsqueeze(-2);  // add before D axis
    
            /* ── ensure h is 4-D ─────────────────────────────────── */
            while (h_sq.dim() < 4) h_sq = h_sq.unsqueeze(-3);  // add before D,D
    
            return TensorMatHyperDual(r_sq, d_sq, h_sq);
        }
    
    
    
    
         
        // ─────────────────────────────────────────────────────────────
        // Remove a singleton row or column (dim = 0 or 1), preserving
        // required ranks: r→2-D, d→3-D, h→4-D.
        [[nodiscard]] TensorMatHyperDual squeeze(int dim) const {
            /* --- normalise dim, allow negatives like PyTorch -------- */
            if (dim < 0) dim += r.dim();          // r.dim() == 2
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("squeeze(): dim must be 0 or 1.");
    
            /* --- size-1 check -------------------------------------- */
            if (r.size(dim) != 1)
                throw std::invalid_argument("squeeze(): chosen dimension is not size-1.");
    
            /* --- squeeze that axis in all parts -------------------- */
            torch::Tensor r_sq = r.squeeze(dim);  // rank ↓ 1
            torch::Tensor d_sq = d.squeeze(dim);  // rank ↓ 1
            torch::Tensor h_sq = h.squeeze(dim);  // rank ↓ 1
    
            /* --- re-add axes to keep invariant ranks --------------- */
            while (r_sq.dim() < 2) r_sq = r_sq.unsqueeze(-1);    // ensure 2-D
            while (d_sq.dim() < 3) d_sq = d_sq.unsqueeze(-2);    // ensure 3-D
            while (h_sq.dim() < 4) h_sq = h_sq.unsqueeze(-3);    // ensure 4-D
    
            return TensorMatHyperDual(r_sq, d_sq, h_sq);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Ensure each tensor is laid out contiguously in memory.
        [[nodiscard]] inline TensorMatHyperDual contiguous() const {
            return TensorMatHyperDual(
                r.is_contiguous() ? r : r.contiguous(),
                d.is_contiguous() ? d : d.contiguous(),
                h.is_contiguous() ? h : h.contiguous());
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise square root with first- and second-order terms.
        //
        //   rₒ = √r
        //   dₒ = d / (2 √r)
        //   hₒ = h /(2 √r)  − (d ⊗ d) /(4 r^{3/2})
        //
        [[nodiscard]] TensorMatHyperDual sqrt() const {
            /* ── promote to complex if any negatives (real dtype) ─── */
            TensorMatHyperDual src =
                (r.is_complex() || torch::all(r >= 0).item<bool>())
                ? *this
                : this->complex();
    
            /* ── real part ────────────────────────────────────────── */
            torch::Tensor r_sqrt = torch::sqrt(src.r);               // [N,L]
            r_sqrt = torch::where(r_sqrt == 0, r_sqrt + 1e-12, r_sqrt);
    
            /* ── dual part  : d /(2 √r) ───────────────────────────── */
            torch::Tensor coeff_dual = 0.5 / r_sqrt;                 // [N,L]
            torch::Tensor d_out = coeff_dual.unsqueeze(-1) * src.d;  // [N,L,1] * [N,L,D]
    
            /* ── hyperdual part ───────────────────────────────────── */
            torch::Tensor coeff_h    = coeff_dual.unsqueeze(-1).unsqueeze(-1);   // [N,L,1,1]
            torch::Tensor coeff_dd   = 0.25 / r_sqrt.pow(3)                      // 1/(4 r^{3/2})
                                        .unsqueeze(-1).unsqueeze(-1);         // [N,L,1,1]
    
            // outer product d⊗d  → [N,L,D,D]
            torch::Tensor dd_outer = src.d.unsqueeze(-1) * src.d.unsqueeze(-2);
    
            torch::Tensor h_out = coeff_h * src.h - coeff_dd * dd_outer;         // [N,L,D,D]
    
            return TensorMatHyperDual(r_sqrt, d_out, h_out);
        }
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Build a TensorMatHyperDual with given real part and zero-filled
        // dual / hyperdual blocks of size `ddim`.
        static TensorMatHyperDual createZero(const torch::Tensor& r, int ddim) {
            // ── validation ──────────────────────────────────────────
            if (!r.defined() || r.dim() != 2)
                throw std::invalid_argument("createZero(): real tensor must be 2-D [N,L].");
            if (ddim <= 0)
                throw std::invalid_argument("createZero(): `ddim` must be positive.");
    
            // ── shapes ─────────────────────────────────────────────
            std::vector<int64_t> dshape = { r.size(0), r.size(1), ddim };          // [N,L,D]
            std::vector<int64_t> hshape = { r.size(0), r.size(1), ddim, ddim };    // [N,L,D,D]
    
            // ── zero tensors with same dtype / device ──────────────
            auto opts = torch::TensorOptions().dtype(r.dtype()).device(r.device());
            torch::Tensor d = torch::zeros(dshape, opts);
            torch::Tensor h = torch::zeros(hshape, opts);
    
            return TensorMatHyperDual(r, d, h);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Produce a TensorMatHyperDual whose components are all-zero and
        // whose shapes/dtype/device follow `x` (for r) and *this* (for D).
        //
        //   • `x` must be 2-D [N,L] and live on the desired dtype/device.
        //   • The dual size D is taken from *this → d.size(2)*.
        //
        [[nodiscard]] TensorMatHyperDual zeros_like(const torch::Tensor& x) const {
            // ── validation ──────────────────────────────────────────
            if (!x.defined() || x.dim() != 2)
                throw std::invalid_argument("zeros_like(): `x` must be a defined 2-D tensor [N,L].");
            if (!d.defined() || d.dim() != 3 || !h.defined() || h.dim() != 4)
                throw std::runtime_error("zeros_like(): dual/hyperdual tensors have wrong rank.");
    
            // ── dimensions ─────────────────────────────────────────
            const auto N = x.size(0);
            const auto L = x.size(1);
            const auto D = d.size(2);             // use existing dual dimension
    
            // ── build zero tensors on x’s dtype/device ─────────────
            auto opts = torch::TensorOptions().dtype(x.dtype()).device(x.device());
            torch::Tensor rc = torch::zeros_like(x);                   // [N,L]
            torch::Tensor dc = torch::zeros({N, L, D},       opts);    // [N,L,D]
            torch::Tensor hc = torch::zeros({N, L, D, D},    opts);    // [N,L,D,D]
    
            return TensorMatHyperDual(rc, dc, hc);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Deep copy (independent storage for all three parts).
        [[nodiscard]] inline TensorMatHyperDual clone() const {
            if (!r.defined() || !d.defined() || !h.defined())
                throw std::runtime_error("TensorMatHyperDual::clone(): undefined tensors.");
    
            return TensorMatHyperDual(r.clone(),          // [N,L]
                                    d.clone(),          // [N,L,D]
                                    h.clone());         // [N,L,D,D]
        }
    
        // ─────────────────────────────────────────────────────────────
        // Concatenate two TensorMatHyperDuals along a chosen axis.
        //
        //   • dim = 0  rows   : stack matrices vertically
        //   • dim = 1  cols   : stack matrices horizontally
        //   • dim = 2  dual   : keep matrix, concatenate sensitivities
        //
        // Throws if the untouched dimensions disagree.
        static TensorMatHyperDual cat(const TensorMatHyperDual &A,
                                      const TensorMatHyperDual &B,
                                      int dim = 2)
        {
            if (dim < 0)
                dim += 4; // allow negative index
            if (dim < 0 || dim > 3)
                throw std::invalid_argument("cat(): dim must be 0,1,2 or 3.");
    
            // helper λ to check shape compatibility on all but `dim`
            auto same_on_other_axes = [dim](const torch::Tensor &x,
                                            const torch::Tensor &y) -> bool
            {
                if (x.dim() != y.dim())
                    return false;
                for (int i = 0; i < x.dim(); ++i)
                    if (i != dim && x.size(i) != y.size(i))
                        return false;
                return true;
            };
    
            // matrix (r) must match exactly when joining on dual axis
            if (dim == 2 && A.r.sizes() != B.r.sizes())
                throw std::invalid_argument("cat(): real parts must match when dim==2.");
    
            // verify all parts are compatible
            if (!same_on_other_axes(A.r, B.r) ||
                !same_on_other_axes(A.d, B.d) ||
                !same_on_other_axes(A.h, B.h))
                throw std::invalid_argument("cat(): tensor shapes mismatch on non-concatenated axes.");
    
            // concatenate
            torch::Tensor r_out = (dim == 2) ? A.r.clone() // keep one copy of r
                                             : torch::cat({A.r, B.r}, dim);
    
            torch::Tensor d_out = torch::cat({A.d, B.d}, dim); // [N,L,D] – dim may be 0,1,2
            torch::Tensor h_out = torch::cat({A.h, B.h}, dim); // [N,L,D,D] – dim 0,1,2, or 3
    
            return TensorMatHyperDual(r_out, d_out, h_out);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Concatenate (TensorMatHyperDual , Tensor)  along the column axis.
        //
        //  • t1.r : [N, L]
        //  • t2   : [N, K]   or  [K]  (broadcasted down the rows)
        //  → result.r : [N, L+K]
        //    result.d : [N, L+K, D]      (zeros for the new cols)
        //    result.h : [N, L+K, D, D]   (zeros for the new cols)
        //
        static TensorMatHyperDual cat(const TensorMatHyperDual &t1,
                                      const torch::Tensor &t2)
        {
            /* ── validate / broadcast t2 to [N,K] ─────────────────── */
            if (!t2.defined())
                throw std::invalid_argument("cat(): t2 is undefined.");
    
            const auto N = t1.r.size(0);
            //const auto L = t1.r.size(1);
            const auto D = t1.d.size(2);
    
            torch::Tensor t2_mat;
            if (t2.dim() == 1)
            {                                             // shape [K] → broadcast rows
                t2_mat = t2.unsqueeze(0).expand({N, -1}); // [N,K]
            }
            else if (t2.dim() == 2)
            {
                if (t2.size(0) != N)
                    throw std::invalid_argument("cat(): t2 rows must match N.");
                t2_mat = t2; // [N,K]
            }
            else
            {
                throw std::invalid_argument("cat(): t2 must be 1-D or 2-D.");
            }
    
            t2_mat = t2_mat.to(t1.r.device()); // move to same device
            const auto K = t2_mat.size(1);
    
            /* ── concatenate real part ────────────────────────────── */
            torch::Tensor r_cat = torch::cat({t1.r, t2_mat}, 1); // [N, L+K]
    
            /* ── zero dual / hyperdual for new cols ───────────────── */
            auto opts = t1.d.options();
            torch::Tensor d_zeros = torch::zeros({N, K, D}, opts);    // [N,K,D]
            torch::Tensor h_zeros = torch::zeros({N, K, D, D}, opts); // [N,K,D,D]
    
            torch::Tensor d_cat = torch::cat({t1.d, d_zeros}, 1); // [N, L+K, D]
            torch::Tensor h_cat = torch::cat({t1.h, h_zeros}, 1); // [N, L+K, D, D]
    
            return TensorMatHyperDual(r_cat, d_cat, h_cat);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise addition of two TensorMatHyperDuals
        [[nodiscard]] inline TensorMatHyperDual operator+(const TensorMatHyperDual& other) const {
            // basic sanity checks
            if (!r.defined() || !d.defined() || !h.defined() ||
                !other.r.defined() || !other.d.defined() || !other.h.defined())
                throw std::invalid_argument("TensorMatHyperDual::operator+: undefined tensors.");
    
            if (r.device() != other.r.device() || d.device() != other.d.device() || h.device() != other.h.device())
                throw std::invalid_argument("TensorMatHyperDual::operator+: tensors must be on the same device.");
    
            if (r.dtype()  != other.r.dtype()  || d.dtype()  != other.d.dtype()  || h.dtype()  != other.h.dtype())
                throw std::invalid_argument("TensorMatHyperDual::operator+: dtype mismatch.");
    
            if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes() || h.sizes() != other.h.sizes())
                throw std::invalid_argument("TensorMatHyperDual::operator+: shape mismatch.");
    
            return TensorMatHyperDual(r + other.r,     // real  [N,L]
                                    d + other.d,     // dual  [N,L,D]
                                    h + other.h);    // hyper [N,L,D,D]
        }
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise addition  TensorMatHyperDual + TensorHyperDual
        //
        //   • this.r : [N, L]
        //   •  oth.r : [N]   or [N,1]               (row vector)
        //            → broadcast across L columns
        //
        //   • this.d : [N, L, D]
        //   •  oth.d : [N, D]  or [N,1,D]           (row vector of duals)
        //
        //   • this.h : [N, L, D, D]
        //   •  oth.h : [N, D, D]  or [N,1,D,D]
        //
        [[nodiscard]] TensorMatHyperDual operator+(const TensorHyperDual& oth) const
        {
            /* ── basic checks ─────────────────────────────────────── */
            if (!r.defined() || !d.defined() || !h.defined() ||
                !oth.r.defined() || !oth.d.defined() || !oth.h.defined())
                throw std::invalid_argument("operator+: undefined tensors.");
    
            if (r.device() != oth.r.device() || d.device() != oth.d.device() || h.device() != oth.h.device())
                throw std::invalid_argument("operator+: tensors must be on the same device.");
    
            if (r.dtype()  != oth.r.dtype() || d.dtype() != oth.d.dtype() || h.dtype() != oth.h.dtype())
                throw std::invalid_argument("operator+: dtype mismatch.");
    
            const auto N = r.size(0);
            const auto L = r.size(1);
            const auto D = d.size(2);
    
            /* ── bring TensorHyperDual to broadcastable shapes ────── */
            // real
            torch::Tensor r_vec = (oth.r.dim() == 1) ? oth.r.unsqueeze(1)  // [N,1]
                                                    : oth.r;             // [N,1] already
    
            // dual
            torch::Tensor d_vec = (oth.d.dim() == 2) ? oth.d.unsqueeze(1)  // [N,1,D]
                                                    : oth.d;             // [N,1,D]
    
            // hyperdual
            torch::Tensor h_vec = (oth.h.dim() == 3) ? oth.h.unsqueeze(1)  // [N,1,D,D]
                                                    : oth.h;             // [N,1,D,D]
    
            /* ── dimension checks ─────────────────────────────────── */
            if (r_vec.size(0) != N ||
                d_vec.size(0) != N || d_vec.size(2) != D ||
                h_vec.size(0) != N || h_vec.size(2) != D || h_vec.size(3) != D)
                throw std::invalid_argument("operator+: shape mismatch between TensorMatHyperDual and TensorHyperDual.");
    
            /* ── broadcast to matrix width L and add ──────────────── */
            torch::Tensor r_sum = r + r_vec.expand({N, L});               // [N,L]
            torch::Tensor d_sum = d + d_vec.expand({N, L, D});            // [N,L,D]
            torch::Tensor h_sum = h + h_vec.expand({N, L, D, D});         // [N,L,D,D]
    
            return TensorMatHyperDual(r_sum, d_sum, h_sum);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Add a scalar to every entry of the real part.
        // Dual and hyperdual sensitivities are unchanged.
        [[nodiscard]] inline TensorMatHyperDual operator+(double scalar) const {
            if (!r.defined())
                throw std::invalid_argument("TensorMatHyperDual::operator+: real tensor is undefined.");
            if (!r.is_floating_point())
                throw std::invalid_argument("TensorMatHyperDual::operator+: real tensor must be floating-point.");
    
            return TensorMatHyperDual(r + scalar,   // shift real part
                                    d,            // dual unchanged
                                    h);           // hyperdual unchanged
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise subtraction of two TensorMatHyperDuals
        [[nodiscard]] inline TensorMatHyperDual operator-(const TensorMatHyperDual& other) const {
            // Basic sanity checks
            if (!r.defined() || !d.defined() || !h.defined() ||
                !other.r.defined() || !other.d.defined() || !other.h.defined())
                throw std::invalid_argument("TensorMatHyperDual::operator- : undefined tensors.");
    
            if (r.device() != other.r.device() ||
                d.device() != other.d.device() ||
                h.device() != other.h.device())
                throw std::invalid_argument("TensorMatHyperDual::operator- : tensors must be on the same device.");
    
            if (r.dtype()  != other.r.dtype() ||
                d.dtype()  != other.d.dtype() ||
                h.dtype()  != other.h.dtype())
                throw std::invalid_argument("TensorMatHyperDual::operator- : dtype mismatch.");
    
            if (r.sizes() != other.r.sizes() ||
                d.sizes() != other.d.sizes() ||
                h.sizes() != other.h.sizes())
                throw std::invalid_argument("TensorMatHyperDual::operator- : shape mismatch.");
    
            return TensorMatHyperDual(r - other.r,     // real   [N,L]
                                    d - other.d,     // dual   [N,L,D]
                                    h - other.h);    // hyper  [N,L,D,D]
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Subtract a scalar from every element of the real part.
        // Dual and hyperdual tensors are unchanged.
        [[nodiscard]] inline TensorMatHyperDual operator-(double scalar) const {
            if (!r.defined())
                throw std::invalid_argument("TensorMatHyperDual::operator-(scalar): real tensor is undefined.");
            if (!r.is_floating_point())
                throw std::invalid_argument("TensorMatHyperDual::operator-(scalar): real tensor must be floating-point.");
    
            return TensorMatHyperDual(r - scalar,   // shift real part
                                    d,            // dual unchanged
                                    h);           // hyperdual unchanged
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise equality test on the real parts.
        // Dual and hyperdual components are ignored.
        [[nodiscard]] inline torch::Tensor operator==(const TensorMatHyperDual& other) const {
            if (!r.defined() || !other.r.defined())
                throw std::invalid_argument("TensorMatHyperDual::operator== : real tensors are undefined.");
            if (r.sizes() != other.r.sizes())
                throw std::invalid_argument("TensorMatHyperDual::operator== : shape mismatch.");
    
            // returns a bool tensor of shape [N, L]
            return r == other.r;
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Unary negation:  -(TensorMatHyperDual)
        [[nodiscard]] inline TensorMatHyperDual operator-() const {
            return TensorMatHyperDual(-r,   // real       [N,L]
                                    -d,   // dual       [N,L,D]
                                    -h);  // hyperdual  [N,L,D,D]
        }
    
        // ─────────────────────────────────────────────────────────────
        // Scalar multiplication:  TensorMatHyperDual * c
        [[nodiscard]] inline TensorMatHyperDual operator*(double scalar) const {
            return TensorMatHyperDual(r * scalar,   // real       [N,L]
                                    d * scalar,   // dual       [N,L,D]
                                    h * scalar);  // hyperdual  [N,L,D,D]
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise quotient  this / other
        [[nodiscard]] TensorMatHyperDual
        operator/(const TensorMatHyperDual& other) const
        {
            /* ----- safety checks ------------------------------------------------- */
            if (r.sizes() != other.r.sizes() ||
                d.sizes() != other.d.sizes() ||
                h.sizes() != other.h.sizes())
                throw std::invalid_argument("TensorMatHyperDual::operator/: shape mismatch.");
    
            /* ----- shorthand ----------------------------------------------------- */
            const torch::Tensor& f  = r;             // numerator  (real  part)
            const torch::Tensor& fp = d;             // numerator  (dual  part)
            const torch::Tensor& fpp= h;             // numerator  (hyper part)
    
            const torch::Tensor& g  = other.r;       // denominator (real)
            const torch::Tensor& gp = other.d;       // denominator (dual)
            const torch::Tensor& gpp= other.h;       // denominator (hyper)
    
            /* ----- avoid divide-by-zero ----------------------------------------- */
            torch::Tensor g_safe = torch::where(g == 0, g + 1e-12, g);
    
            /* ----- useful powers ------------------------------------------------- */
            torch::Tensor g_inv   = 1.0 / g_safe;                // g⁻¹   [N,L]
            torch::Tensor g_inv2  = g_inv * g_inv;               // g⁻²   [N,L]
            torch::Tensor g_inv3  = g_inv2 * g_inv;              // g⁻³   [N,L]
    
            /* -------------------------------------------------------------------- *
            * real part                                                            *
            * -------------------------------------------------------------------- */
            torch::Tensor r_out = f * g_inv;                     // [N,L]
    
            /* -------------------------------------------------------------------- *
            * dual part  dq = (fp·g − f·gp) / g²                                   *
            * -------------------------------------------------------------------- */
            torch::Tensor d_out =
                (fp * g.unsqueeze(-1) -                       // fp·g
                f.unsqueeze(-1) * gp) * g_inv2.unsqueeze(-1);  // /g²
                // shape → [N,L,D]
    
            /* -------------------------------------------------------------------- *
            * hyperdual part                                                       *
            *    h_out = ( fpp·g − 2 sym(fp⊗gp) − f·gpp ) / g²                     *
            *            + 2 f (gp⊗gp)/g³                                          *
            * -------------------------------------------------------------------- */
    
            // outer products:  fp⊗gp  and  gp⊗gp   → [N,L,D,D]
            torch::Tensor fp_gp = fp.unsqueeze(-1) * gp.unsqueeze(-2);
            torch::Tensor gp_gp = gp.unsqueeze(-1) * gp.unsqueeze(-2);
    
            // first bracket
            torch::Tensor term1 =
                fpp * g.unsqueeze(-1).unsqueeze(-1)          // f'' · g
            - 2.0 * fp_gp                                 // –2 fp⊗gp
            - f.unsqueeze(-1).unsqueeze(-1) * gpp;        // – f · g''
    
            term1 *= g_inv2.unsqueeze(-1).unsqueeze(-1);     // /g²
    
            // second bracket
            torch::Tensor term2 =
                2.0 * f.unsqueeze(-1).unsqueeze(-1) * gp_gp * g_inv3.unsqueeze(-1).unsqueeze(-1);  // 2 f (gp⊗gp)/g³
    
            torch::Tensor h_out = term1 + term2;             // [N,L,D,D]
    
            return TensorMatHyperDual(r_out, d_out, h_out);
        }
    
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise division:  TensorMatHyperDual  /  TensorHyperDual
        //
        //   • this.r : [N, L]
        //   •  oth.r : [N]      or [N,1]               (row vector)
        //     → broadcast across L columns
        //
        // The helper constructor  TensorMatHyperDual( TensorDual, dim=1 )
        // adds a singleton column axis, giving shapes
        //   r : [N,1] , d : [N,1,D] , h : [N,1,D,D]
        //
        [[nodiscard]] TensorMatHyperDual operator/(const TensorHyperDual& oth) const
        {
            // build a [N,1] matrix view of `oth`
            TensorMatHyperDual divisor(
                oth.r.unsqueeze(1),     // (N)   -> (N,1)
                oth.d.unsqueeze(1),     // (N,D) -> (N,1,D)
                oth.h.unsqueeze(1));    // (N,D,D) -> (N,1,D,D)
        
            // now re-use mat ÷ mat implementation (broadcasted over columns)
            return *this / divisor;
        }
        
    
        
        // ─────────────────────────────────────────────────────────────
        // Element-wise division by a plain tensor.
        //
        //  • `other` may be
        //        ()         – scalar
        //        [N]        – row vector
        //        [N, L]     – full matrix
        //
        //  Shapes are broadcast to [N, L].  The dual axis D is broadcast
        //  automatically by adding a trailing singleton.
        [[nodiscard]] TensorMatHyperDual operator/(const torch::Tensor& other) const
        {
            /* ── validation ───────────────────────────────────────── */
            if (!other.defined())
                throw std::invalid_argument("Tensor divisor is undefined.");
            if (other.is_complex() != r.is_complex())
                throw std::invalid_argument("Divisor must have the same real/complex dtype.");
    
            // Accept (), [N], or [N,L]; rely on PyTorch’s broadcast rules.
            auto divisor = other.to(r.device());
    
            // Guard against zeros
            if (torch::any(divisor == 0).item<bool>())
                throw std::runtime_error("Division by zero in tensor divisor.");
    
            /* ── real part ────────────────────────────────────────── */
            torch::Tensor r_out = r / divisor;                     // broadcast → [N,L]
    
            /* ── dual part : scale by divisor ────────────────────── */
            torch::Tensor d_out = d / divisor.unsqueeze(-1);       // [N,L,1] -> [N,L,D]
    
            /* ── hyper-dual part : scale by divisor ─────────────── */
            torch::Tensor h_out = h / divisor.unsqueeze(-1)
                                        .unsqueeze(-1);         // [N,L,1,1]
    
            return TensorMatHyperDual(r_out, d_out, h_out);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Divide every component by a scalar.
        // Dual and hyper-dual parts scale linearly.
        [[nodiscard]] inline TensorMatHyperDual operator/(double scalar) const {
            if (scalar == 0.0)
                throw std::invalid_argument("TensorMatHyperDual::operator/(scalar): division by zero.");
    
            return TensorMatHyperDual(r / scalar,   // real       [N,L]
                                    d / scalar,   // dual       [N,L,D]
                                    h / scalar);  // hyperdual  [N,L,D,D]
        }
    
        // ─────────────────────────────────────────────────────────────
        // Advanced slicing on the matrix axes (rows / cols).
        // Keeps ranks:   r → 2-D ,  d → 3-D ,  h → 4-D.
        //
        [[nodiscard]]
        TensorMatHyperDual index(const std::vector<torch::indexing::TensorIndex>& idx) const
        {
            using torch::indexing::Slice;
    
            /* ── validate indices ─────────────────────────────────── */
            if (idx.size() > 2)
                throw std::invalid_argument("index(): at most two indices (rows, cols) allowed.");
    
            for (const auto& t : idx)
                if (t.is_integer())
                    throw std::invalid_argument("index(): integer indices would drop a dimension; use Slice() instead.");
    
            /* ── real part ────────────────────────────────────────── */
            torch::Tensor r_sub = r.index(idx);          // rank ≤ 2
            while (r_sub.dim() < 2) r_sub = r_sub.unsqueeze(-1);   // ensure 2-D
    
            /* ── dual part ────────────────────────────────────────── */
            auto idx_d = idx;        // copy row / col selectors
            idx_d.push_back(Slice());                // keep all D
            torch::Tensor d_sub = d.index(idx_d);    // rank ≤ 3
            while (d_sub.dim() < 3) d_sub = d_sub.unsqueeze(-2);   // ensure 3-D
    
            /* ── hyperdual part ───────────────────────────────────── */
            auto idx_h = idx;
            idx_h.push_back(Slice());                // keep first D
            idx_h.push_back(Slice());                // keep second D
            torch::Tensor h_sub = h.index(idx_h);    // rank ≤ 4
            while (h_sub.dim() < 4) h_sub = h_sub.unsqueeze(-3);   // ensure 4-D
    
            return TensorMatHyperDual(r_sub, d_sub, h_sub);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Select a single row (first axis) without collapsing rank.
        //
        // Result shapes
        //   r : [1, L]
        //   d : [1, L, D]
        //   h : [1, L, D, D]
        //
        [[nodiscard]] TensorMatHyperDual index(int idx) const {
            if (idx < 0 || idx >= r.size(0))
                throw std::out_of_range("TensorMatHyperDual::index(int): row index out of bounds.");
    
            torch::Tensor r_row = r.index({ idx }).unsqueeze(0);        // [1,L]
            torch::Tensor d_row = d.index({ idx }).unsqueeze(0);        // [1,L,D]
            torch::Tensor h_row = h.index({ idx }).unsqueeze(0);        // [1,L,D,D]
    
            return TensorMatHyperDual(r_row, d_row, h_row);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Row-mask selection with a 1-D boolean tensor.
        //
        //   mask length must equal N (row count).
        //   Result shapes: r [M,L] , d [M,L,D] , h [M,L,D,D]
        //   where M = mask.sum().
        [[nodiscard]] TensorMatHyperDual index(const torch::Tensor& mask) const {
            /* ── validate mask ────────────────────────────────────── */
            if (!mask.defined() || mask.scalar_type() != torch::kBool)
                throw std::invalid_argument("index(mask): mask must be a defined bool tensor.");
            if (mask.dim() != 1 || mask.size(0) != r.size(0))
                throw std::invalid_argument("index(mask): mask must be 1-D with length N.");
            if (mask.device() != r.device())
                throw std::invalid_argument("index(mask): mask must be on the same device as tensors.");
    
            /* ── select rows ──────────────────────────────────────── */
            using torch::indexing::Slice;
    
            torch::Tensor r_sel = r.index({ mask, Slice() });                    // [M,L]
            torch::Tensor d_sel = d.index({ mask, Slice(), Slice() });           // [M,L,D]
            torch::Tensor h_sel = h.index({ mask, Slice(), Slice(), Slice() });  // [M,L,D,D]
    
            return TensorMatHyperDual(r_sel, d_sel, h_sel);
        }
    
        // ─────────────────────────────────────────────────────────────
        // einsum( TensorMatHyperDual , TensorHyperDual ) → TensorHyperDual
        // Limited to two operands; the user’s equation must *not* contain
        // the reserved labels 'z' or 'w'.
        static TensorHyperDual einsum(const std::string &eq,
                                      const TensorMatHyperDual &A,
                                      const TensorHyperDual &b)
        {
            /* ---------- basic validation -------------------------------------- */
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            if (eq.find('z') != std::string::npos || eq.find('w') != std::string::npos)
                throw std::invalid_argument("einsum string must not contain reserved labels 'z' or 'w'.");
    
            if (!A.r.defined() || !b.r.defined())
                throw std::invalid_argument("Input tensors must be defined and non-empty.");
    
            /* ---------- split equation ---------------------------------------- */
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            /* ---------- real contraction ------------------------------------- */
            torch::Tensor r_out = torch::einsum(eq, {A.r, b.r});
    
            /* ---------- first-order term  (product rule) ---------------------- */
            std::string eq_dA = subA + "z," + subB + " -> " + subO + "z";
            std::string eq_dB = subA + "," + subB + "z -> " + subO + "z";
    
            torch::Tensor dA = torch::einsum(eq_dA, {A.d, b.r});
            torch::Tensor dB = torch::einsum(eq_dB, {A.r, b.d});
    
            torch::Tensor d_out = dA + dB; // [*,*,D]
    
            /* ---------- second-order term ------------------------------------- */
            // d⊗d  term
            std::string eq_dd = subA + "z," + subB + "w -> " + subO + "zw";
            torch::Tensor dd = torch::einsum(eq_dd, {A.d, b.d});
    
            // A.h term
            std::string eq_hA = subA + "zw," + subB + " -> " + subO + "zw";
            torch::Tensor hA = torch::einsum(eq_hA, {A.h, b.r});
    
            // b.h term
            std::string eq_hB = subA + " ," + subB + "zw -> " + subO + "zw";
            torch::Tensor hB = torch::einsum(eq_hB, {A.r, b.h});
    
            torch::Tensor h_out = dd + hA + hB; // [*,*,D,D]
    
            return TensorHyperDual(std::move(r_out),
                                   std::move(d_out),
                                   std::move(h_out));
        }
    
        // ─────────────────────────────────────────────────────────────
        // einsum( TensorMatHyperDual , Tensor ) → TensorMatHyperDual
        // Two operands only.  The user’s equation must NOT contain ‘z’ or ‘w’.
        static TensorMatHyperDual einsum(const std::string &eq,
                                         const TensorMatHyperDual &A,
                                         const torch::Tensor &B)
        {
            /* ── sanity checks ────────────────────────────────────── */
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            if (eq.find('z') != std::string::npos || eq.find('w') != std::string::npos)
                throw std::invalid_argument("einsum string may not contain reserved labels 'z' or 'w'.");
    
            if (!A.r.defined() || !B.defined())
                throw std::invalid_argument("Input tensors must be defined.");
    
            /* ── split equation:  subA , subB -> subO ─────────────── */
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            /* ── real contraction ─────────────────────────────────── */
            torch::Tensor r_out = torch::einsum(eq, {A.r, B});
    
            /* ── dual contraction: A.d carries ‘z’ ────────────────── */
            std::string eq_d = subA + "z," + subB + " -> " + subO + "z";
            torch::Tensor d_out = torch::einsum(eq_d, {A.d, B}); // [*,*,D]
    
            /* ── hyper contraction: A.h carries ‘z w’ ─────────────── */
            std::string eq_h = subA + "zw," + subB + " -> " + subO + "zw";
            torch::Tensor h_out = torch::einsum(eq_h, {A.h, B}); // [*,*,D,D]
    
            return TensorMatHyperDual(std::move(r_out),
                                      std::move(d_out),
                                      std::move(h_out));
        }
    
        // ─────────────────────────────────────────────────────────────
        // einsum( TensorHyperDual , TensorMatHyperDual ) -> TensorHyperDual
        // Two operands only; 'z' and 'w' are reserved for sensitivities.
        static TensorHyperDual einsum(const std::string &eq,
                                      const TensorHyperDual &a,
                                      const TensorMatHyperDual &B)
        {
            /* ---------- validate equation ------------------------------------ */
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
            if (eq.find('z') != std::string::npos || eq.find('w') != std::string::npos)
                throw std::invalid_argument("einsum string may not contain reserved labels 'z' or 'w'.");
    
            /* ---------- split subscripts ------------------------------------- */
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            /* ---------- real part ------------------------------------------- */
            torch::Tensor r_out = torch::einsum(eq, {a.r, B.r});
    
            /* ---------- first-order term  (product rule) --------------------- */
            std::string eq_da = subA + "z," + subB + " -> " + subO + "z";
            std::string eq_db = subA + "," + subB + "z -> " + subO + "z";
    
            torch::Tensor d_out =
                torch::einsum(eq_da, {a.d, B.r})    // a' · B
                + torch::einsum(eq_db, {a.r, B.d}); // a  · B'
    
            /* ---------- second-order term ------------------------------------ */
            std::string eq_dd = subA + "z," + subB + "w -> " + subO + "zw";
            std::string eq_hA = subA + "zw," + subB + " -> " + subO + "zw";
            std::string eq_hB = subA + "," + subB + "zw -> " + subO + "zw";
    
            torch::Tensor h_out =
                torch::einsum(eq_dd, {a.d, B.d})    // a'⊗B'
                + torch::einsum(eq_hA, {a.h, B.r})  // a''·B
                + torch::einsum(eq_hB, {a.r, B.h}); // a ·B''
    
            return TensorHyperDual(std::move(r_out),
                                   std::move(d_out),
                                   std::move(h_out));
        }
    
        // ─────────────────────────────────────────────────────────────
        // einsum( A , B )   where A, B are TensorMatHyperDual
        //
        // • A.r , B.r : [N,L]     (matrix values)
        // • A.d , B.d : [N,L,D]   (first-order)
        // • A.h , B.h : [N,L,D,D] (second-order)
        //
        // Result: TensorMatHyperDual whose shapes follow the user’s
        //         subscripts; ‘z’ is appended for dual, ‘zw’ for hyper.
        //
        static TensorMatHyperDual einsum(const std::string &eq,
                                         const TensorMatHyperDual &A,
                                         const TensorMatHyperDual &B)
        {
            /* ── validate einsum string ──────────────────────────── */
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
            if (eq.find('z') != std::string::npos || eq.find('w') != std::string::npos)
                throw std::invalid_argument("einsum string may not contain reserved labels 'z' or 'w'.");
    
            /* ── split subscripts ───────────────────────────────── */
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            /* ── real contraction ───────────────────────────────── */
            torch::Tensor r_out = torch::einsum(eq, {A.r, B.r});
    
            /* ── first-order term (product rule) ────────────────── */
            std::string eq_dA = subA + "z," + subB + " -> " + subO + "z";
            std::string eq_dB = subA + "," + subB + "z -> " + subO + "z";
    
            torch::Tensor d_out =
                torch::einsum(eq_dA, {A.d, B.r})    // A' · B
                + torch::einsum(eq_dB, {A.r, B.d}); // A  · B'
    
            /* ── second-order term ──────────────────────────────── */
            std::string eq_dd = subA + "z," + subB + "w -> " + subO + "zw";
            std::string eq_hA = subA + "zw," + subB + " -> " + subO + "zw";
            std::string eq_hB = subA + "," + subB + "zw -> " + subO + "zw";
    
            torch::Tensor h_out =
                2.0 * torch::einsum(eq_dd, {A.d, B.d}) // 2 A'⊗B'
                + torch::einsum(eq_hA, {A.h, B.r})     // A''·B
                + torch::einsum(eq_hB, {A.r, B.h});    // A ·B''
    
            return TensorMatHyperDual(std::move(r_out),
                                      std::move(d_out),
                                      std::move(h_out));
        }
    
        // ─────────────────────────────────────────────────────────────
        // Masked in-place assignment using a 1-D boolean mask.
        //
        //   • `mask` length N selects rows to replace.
        //   • `value` must have shapes
        //        r : [M,L]       where M = mask.sum()
        //        d : [M,L,D]
        //        h : [M,L,D,D]
        //
        inline void index_put_(const torch::Tensor &mask,
                               const TensorMatHyperDual &value)
        {
            /* ── validate mask ────────────────────────────────────── */
            if (!mask.defined() || mask.scalar_type() != torch::kBool)
                throw std::invalid_argument("index_put_: mask must be a defined bool tensor.");
            if (mask.dim() != 1 || mask.size(0) != r.size(0))
                throw std::invalid_argument("index_put_: mask must be 1-D with length N.");
            if (mask.device() != r.device())
                throw std::invalid_argument("index_put_: mask must be on the same device as tensors.");
    
            /* ── validate value shapes ────────────────────────────── */
            auto M = mask.sum().item<int64_t>(); // rows being written
            if (value.r.size(0) != M || value.r.size(1) != r.size(1) ||
                value.d.sizes() != torch::IntArrayRef({M, r.size(1), d.size(2)}) ||
                value.h.sizes() != torch::IntArrayRef({M, r.size(1), h.size(2), h.size(3)}))
                throw std::invalid_argument("index_put_: value shapes incompatible with masked slice.");
    
            using torch::indexing::Slice;
    
            /* ── write real part ──────────────────────────────────── */
            r.index_put_({mask, Slice()}, value.r); // rows, all cols
    
            /* ── write dual part ──────────────────────────────────── */
            d.index_put_({mask, Slice(), Slice()}, value.d); // rows, all cols, all D
    
            /* ── write hyper-dual part ────────────────────────────── */
            h.index_put_({mask, Slice(), Slice(), Slice()}, value.h); // rows, all cols, D, D
        }
    
        // ─────────────────────────────────────────────────────────────
        // In-place assignment on the **column axis** using a single
        // TensorIndex (`mask`) that selects one or more columns.
        //
        //   • mask may be Slice(), IntArrayRef, or a single integer.
        //   • `value` must have exactly the same number of rows N and
        //     the same dual size D, but may have 1 or more columns
        //     matching mask’s length K.
        //
        // Shapes accepted for `value`
        //   r : [N,K]          d : [N,K,D]          h : [N,K,D,D]
        //
        inline void index_put_(const torch::indexing::TensorIndex &mask,
                               const TensorMatHyperDual &value)
        {
            using namespace torch::indexing; // Slice, None, …

            const auto N = r.size(0); // rows
            const auto L = r.size(1); // columns
            const auto D = d.size(2); // dual width

            /* --------------------------------------------------------------
            1.  How many columns (K) does ‘mask’ refer to?
            Instead of deciphering Slice / SymInt internals, ask
            PyTorch to index a dummy 1-D tensor and take its length.
            ----------------------------------------------------------------*/
            int64_t K;
            if (mask.is_none())
            { //  A[:, :]   («:» => all)
                K = L;
            }
            else
            { //  integer, slice, list, …
                // dummy 1-D vector  [0, 1, …, L-1]   — lives on CPU, int64
                auto dummy = torch::arange(L, torch::kLong);
                auto picked = dummy.index({mask}); // result shape [K]
                K = picked.size(0);
            }

            /* --------------------------------------------------------------
            2.  Shape sanity for the replacement block
            r : [N,K]     d : [N,K,D]     h : [N,K,D,D]
            ----------------------------------------------------------------*/
            TORCH_CHECK(value.r.sizes() == torch::IntArrayRef({N, K}) &&
                            value.d.sizes() == torch::IntArrayRef({N, K, D}) &&
                            value.h.sizes() == torch::IntArrayRef({N, K, D, D}),
                        "index_put_: value tensors have incompatible shapes");

            /* --------------------------------------------------------------
            3.  Write the new data
            ----------------------------------------------------------------*/
            r.index_put_({Slice(), mask}, value.r);                   // (N,K)
            d.index_put_({Slice(), mask, Slice()}, value.d);          // (N,K,D)
            h.index_put_({Slice(), mask, Slice(), Slice()}, value.h); // (N,K,D,D)
        }

        // ─────────────────────────────────────────────────────────────
        // Masked assignment with an ordinary tensor.
        //
        //   • `mask` may specify rows and/or columns (≤2 entries).
        //   • `value` must match the shape of r.index(mask).
        //   • The affected dual / hyper-dual entries are zeroed.
        //
        inline void index_put_(const std::vector<torch::indexing::TensorIndex> &mask,
                               const torch::Tensor &value)
        {
            using torch::indexing::Slice;
    
            /* ── shape check for the real part ───────────────────── */
            auto r_slice = r.index(mask); // view
            if (value.sizes() != r_slice.sizes())
                throw std::invalid_argument(
                    "index_put_: `value` shape must match selected real slice.");
    
            /* ── write real matrix --------------------------------- */
            r.index_put_(mask, value);
    
            /* ── build masks that keep the sensitivity axes -------- */
            std::vector<torch::indexing::TensorIndex> mask_d = mask;
            mask_d.push_back(Slice()); // keep D
    
            std::vector<torch::indexing::TensorIndex> mask_h = mask;
            mask_h.push_back(Slice()); // keep D
            mask_h.push_back(Slice()); // keep D
    
            /* ── zero corresponding dual / hyper-dual entries —— */
            d.index_put_(mask_d, 0.0); // dual  [*,*,D]
            h.index_put_(mask_h, 0.0); // hyper [*,*,D,D]
        }
    };
}