#pragma once
#include <torch/torch.h>


#include <type_traits> // For std::is_scalar
#include <vector>
#include <sstream>  // For std::ostringstream
#include <iomanip>   // for std::setprecision
#include <stdexcept>
#include "tensordual.hpp"
namespace janus {
/**
 * @brief The TensorMatDual class tracks first‑order sensitivities of a matrix.
 *
 * This version stores the real part as a 2‑D tensor of shape **[N, L]** and the dual
 * part as a 3‑D tensor of shape **[N, L, D]**, removing the leading batch dimension
 * that existed in earlier drafts so that it mirrors the layout used by `TensorDual`.
 */
class TensorMatDual : public torch::CustomClassHolder {
    public:
        // ──────────────────────────────────────────────────────────────────────────────
        // Data members
        torch::Tensor r;                    ///< Real part  – shape [N, L]
        torch::Tensor d;                    ///< Dual part  – shape [N, L, D]
        torch::Dtype  dtype_  = torch::kFloat64; ///< Underlying scalar type
        torch::Device device_ = torch::kCPU;     ///< Storage device
    
        // ──────────────────────────────────────────────────────────────────────────────
        // Constructors
    
        /**
         * @brief Construct from real and dual tensors.
         *
         * @param r 2‑D tensor of shape **[N, L]** (real part).
         * @param d 3‑D tensor of shape **[N, L, D]** (dual sensitivities).
         *
         * @throws std::invalid_argument if the dimensionalities or devices mismatch.
         */
        TensorMatDual(torch::Tensor r, torch::Tensor d) {
            if (r.dim() != 2) {
                throw std::invalid_argument("TensorMatDual: real part must be 2‑D [N, L].");
            }
            if (d.dim() != 3) {
                throw std::invalid_argument("TensorMatDual: dual part must be 3‑D [N, L, D].");
            }
            if (r.device() != d.device()) {
                throw std::invalid_argument("TensorMatDual: real and dual tensors must share the same device.");
            }
            if (r.sizes()[0] != d.sizes()[0] || r.sizes()[1] != d.sizes()[1]) {
                throw std::invalid_argument("TensorMatDual: mismatched [N, L] dimensions between real and dual parts.");
            }
    
            this->r = std::move(r);
            this->d = std::move(d);
            this->dtype_  = torch::typeMetaToScalarType(this->r.dtype());
            this->device_ = this->r.device();
        }
    
        /**
         * @brief Default constructor – creates empty tensors on CPU using double precision.
         */
        TensorMatDual()
            : r(torch::empty({0, 0},     torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))),
              d(torch::empty({0, 0, 0},  torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))) {}
    
        /**
         * @brief Deep‑copy constructor.
         */
        TensorMatDual(const TensorMatDual& other) {
            this->r = other.r.clone();
            this->d = other.d.clone();
            this->dtype_  = other.dtype_;
            this->device_ = other.device_;
        }
    
        
        // ──────────────────────────────────────────────────────────────────────────────
        // Assignment operators (rule‑of‑five – defaulted move semantics are fine)
        TensorMatDual& operator=(const TensorMatDual& other) {
            if (this != &other) {
                r = other.r.clone();
                d = other.d.clone();
                dtype_  = other.dtype_;
                device_ = other.device_;
            }
            return *this;
        }
    
        TensorMatDual(TensorMatDual&&)            noexcept = default;
        TensorMatDual& operator=(TensorMatDual&&) noexcept = default;
    
        // ──────────────────────────────────────────────────────────────────────────────
        // Utility helpers
    
        /** @return Number of rows N. */
        inline int64_t rows() const { return r.sizes().empty() ? 0 : r.size(0); }
    
        /** @return Number of columns L. */
        inline int64_t cols() const { return r.sizes().empty() ? 0 : r.size(1); }
    
        /** @return Dual dimension D. */
        inline int64_t dual_dim() const { return d.sizes().empty() ? 0 : d.size(2); }
    
        // ─────────────────────────────────────────────────────────────
        // In–place: move both tensors to a new device.
        inline TensorMatDual& to_(const torch::Device& device) {
            // Fast-return if we’re already on the requested device.
            if (device == device_) return *this;
    
            r = r.to(device);
            d = d.to(device);
            device_ = device;
            return *this;
        }
    
        // ─────────────────────────────────────────────────────────────
        // Query the storage device (CPU / CUDA, etc.)
        [[nodiscard]] inline torch::Device device() const noexcept {
            return device_;
        }
    
        // ─────────────────────────────────────────────────────────────
        // Promote a TensorDual (first-order sensitivities of a vector) to TensorMatDual.
        explicit TensorMatDual(const TensorDual& x, int dim = 2)
        : TensorMatDual(
            /* real */ x.r,                          // shape [N, L]
            /* dual */ ([&] {                       // shape [N, L, 1]
                if (x.r.dim() != 2 || x.d.dim() != 2)
                    throw std::invalid_argument("TensorDual must be 2-D [N,L] to convert to TensorMatDual.");
                if (dim < 0 || dim > 2)
                    throw std::invalid_argument("Invalid unsqueeze dimension (valid: 0,1,2).");
                return x.d.unsqueeze(dim);          // adds the new D-axis
            })()
        )
        {}
    
        // ─────────────────────────────────────────────────────────────
        // Return a complex-valued copy of this TensorMatDual
        [[nodiscard]] TensorMatDual complex() const {
            // Helper: promote a real tensor to complex if needed
            auto promote = [this](const torch::Tensor& t) -> torch::Tensor {
                return t.is_complex() ? t
                                    : torch::complex(t, torch::zeros_like(t)).to(device_);
            };
    
            torch::Tensor rc = promote(r);   // real part  → complex
            torch::Tensor dc = promote(d);   // dual part  → complex
    
            return TensorMatDual(rc, dc);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise complex conjugate (no-op for real tensors)
        [[nodiscard]] inline TensorMatDual conj() const {
            return TensorMatDual(r.conj(), d.conj());
        }
    
        // ─────────────────────────────────────────────────────────────
        // Pretty-print a TensorMatDual
        friend std::ostream& operator<<(std::ostream& os, const TensorMatDual& obj) {
            os << "TensorMatDual {\n"
            << "  dtype   : " << torch::toString(obj.dtype_)  << '\n'
            << "  device  : " << obj.device_                 << '\n'
            << "  r (N,L) : " << obj.r.sizes()               << '\n'
            << "  d (N,L,D): " << obj.d.sizes()              << '\n'
            << "  r = " << obj.r << '\n'
            << "  d = " << obj.d << '\n'
            << "}";
            return os;
        }
    
        // ─────────────────────────────────────────────────────────────
        // Remove size-1 axes while preserving [N,L] / [N,L,D] layout
        [[nodiscard]] TensorMatDual squeeze() const {
            torch::Tensor r_sq = r.squeeze();   // might drop down to 0-,1-,or 2-D
            torch::Tensor d_sq = d.squeeze();   // might drop down to 2- or 3-D
    
            // ── Ensure r stays 2-D  ─────────────────────────────────
            switch (r_sq.dim()) {
                case 0: r_sq = r_sq.unsqueeze(0).unsqueeze(1); break; // scalar → [1,1]
                case 1: r_sq = r_sq.unsqueeze( (r.size(0) == 1) ? 0 : 1 ); break; // [L] or [N]
                case 2: break;                                           // already fine
                default:
                    throw std::runtime_error("TensorMatDual::squeeze(): real part became >2-D");
            }
    
            // ── Ensure d stays 3-D  ─────────────────────────────────
            if (d_sq.dim() == 2) {                 // lost the dual axis
                d_sq = d_sq.unsqueeze(2);          // add it back (D = 1)
            } else if (d_sq.dim() != 3) {
                throw std::runtime_error("TensorMatDual::squeeze(): dual part has invalid rank");
            }
    
            return TensorMatDual(r_sq, d_sq);
        }
    
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Remove a singleton dimension while preserving [N,L] / [N,L,D] layout
        [[nodiscard]] TensorMatDual squeeze(int dim) const {
            if (dim < 0 || dim > 2)
                throw std::invalid_argument("TensorMatDual::squeeze(dim): dim must be 0, 1, or 2.");
    
            // Copy tensors so we don’t mutate *this
            torch::Tensor r_new = r;
            torch::Tensor d_new = d;
    
            if (dim < 2) {
                // Squeezing one of the shared axes (N or L) – must be size-1 in *both* tensors
                if (r.size(dim) != 1)
                    throw std::invalid_argument("Real tensor: chosen dim isn’t size-1, cannot squeeze.");
                if (d.size(dim) != 1)
                    throw std::invalid_argument("Dual tensor: chosen dim isn’t size-1, cannot squeeze.");
    
                r_new = r.squeeze(dim);   // may drop to 1-D
                d_new = d.squeeze(dim);   // stays 3-D or drops to 2-D
            } else { // dim == 2  (dual-only axis D)
                if (d.size(2) != 1)
                    throw std::invalid_argument("Dual tensor: D-axis isn’t size-1, cannot squeeze.");
                d_new = d.squeeze(2);     // drops to 2-D
                // real part unchanged
            }
    
            // ── Restore invariant shapes ────────────────────────────
            if (r_new.dim() == 1) {                // lost N or L axis
                // Reinsert at the same logical position so we’re back to [N,L]
                r_new = (dim == 0) ? r_new.unsqueeze(0)  // squeezed rows → add row axis back
                                : r_new.unsqueeze(1); // squeezed cols → add col axis back
            }
    
            if (d_new.dim() == 2) {
                // D-axis was removed (or fell through); put it back as singleton
                d_new = d_new.unsqueeze(2);
            }
    
            return TensorMatDual(r_new, d_new);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Return a copy whose tensors are laid out contiguously in memory.
        [[nodiscard]] inline TensorMatDual contiguous() const {
            return TensorMatDual(r.contiguous(), d.contiguous());
        }
    
        // ─────────────────────────────────────────────────────────────
        // Return an identity-matrix TensorMatDual (r = I, d = 0)
        //   • r  : shape [N,L]  (identity on the leading min(N,L) diag)
        //   • d  : shape [N,L,D]  (all zeros)
        [[nodiscard]] TensorMatDual eye() const {
            // Sanity-check current layout
            if (r.dim() != 2)
                throw std::runtime_error("TensorMatDual::eye(): real tensor must be 2-D [N,L].");
            if (d.dim() != 3)
                throw std::runtime_error("TensorMatDual::eye(): dual tensor must be 3-D [N,L,D].");
    
            const auto N = r.size(0);
            const auto L = r.size(1);
            const auto D = d.size(2);
    
            // Real part – identity (handles rectangular case by eye(N,L))
            torch::Tensor r_eye = torch::eye(N, L, r.options());
    
            // Dual part – zeros
            torch::Tensor d_zero = torch::zeros({N, L, D}, d.options());
    
            return TensorMatDual(r_eye, d_zero);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Sum along a chosen axis, keeping the reduced dimension (keepdim = true)
        //
        // Valid axes:
        //   0 → rows  (N)
        //   1 → cols  (L)
        //   2 → dual  (D)   — applies only to d; r is left unchanged.
        //
        [[nodiscard]] TensorMatDual sum(int dim) const {
            if (dim < 0 || dim > 2)
                throw std::invalid_argument("TensorMatDual::sum(dim): dim must be 0, 1, or 2.");
    
            torch::Tensor r_sum, d_sum;
    
            if (dim < 2) {                      // shared axes (N or L)
                if (r.dim() != 2 || d.dim() != 3)
                    throw std::runtime_error("TensorMatDual::sum(): tensors are not in [N,L]/[N,L,D] form.");
    
                r_sum = r.sum(dim, /*keepdim=*/true);      // shape preserved: [1,L] or [N,1]
                d_sum = d.sum(dim, /*keepdim=*/true);      // shape preserved: [1,L,D] / [N,1,D]
            } else {                           // dim == 2  (dual axis)
                // Real part unaffected; clone() keeps ownership separate
                r_sum = r.clone();
                d_sum = d.sum(2, /*keepdim=*/true);        // [N,L,1]
            }
    
            return TensorMatDual(r_sum, d_sum);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise square:  r ↦ r² ,  d ↦ 2·r·d
        [[nodiscard]] TensorMatDual square() const {
            // Ensure tensors are in expected ranks
            if (r.dim() != 2 || d.dim() != 3)
                throw std::runtime_error("TensorMatDual::square(): tensors must be [N,L] and [N,L,D].");
    
            // Real part
            torch::Tensor r_sq = r.square();                 // [N,L]
    
            // Dual part: 2 * r * d   (broadcast r along D)
            torch::Tensor d_sq = 2.0 * r.unsqueeze(-1) * d;  // [N,L,1] * [N,L,D] → [N,L,D]
    
            return TensorMatDual(r_sq, d_sq);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise square-root with first-order sensitivities
        //
        //   r_out = sqrt(r)
        //   d_out = d / (2 * sqrt(r))   (chain rule)
        //
        // If r has negative values and is real-typed, we promote to complex first.
        [[nodiscard]] TensorMatDual sqrt() const {
            // Promote to complex if needed for negative arguments
            TensorMatDual src = (r.is_complex() || torch::all(r >= 0).item<bool>())
                                ? *this
                                : this->complex();    // returns complex-valued copy
    
            // Real part √r
            torch::Tensor r_sqrt = src.r.sqrt();       // [N,L]
    
            // Dual part: d / (2√r)
            torch::Tensor coeff = 0.5 / r_sqrt;        // [N,L]
            torch::Tensor d_sqrt = coeff.unsqueeze(-1) * src.d; // broadcast over D → [N,L,D]
    
            return TensorMatDual(r_sqrt, d_sqrt);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Row-wise L2 norm.
        //
        //  • r ∈ ℝ[N,L]               →  ‖r‖₂  ∈ ℝ[N,1]
        //  • d ∈ ℝ[N,L,D]             →  d_out ∈ ℝ[N,1,D]
        //
        // Chain rule:  ∂‖r‖₂/∂x = x / ‖r‖₂   (for each row)
        //
        // If a row-norm is zero we return zero duals for that row so we
        // avoid division-by-zero.
        [[nodiscard]] TensorMatDual normL2() const {
            if (r.dim() != 2 || d.dim() != 3)
                throw std::runtime_error("TensorMatDual::normL2(): tensors must be [N,L] and [N,L,D].");
    
            // 1. row-wise norm, keepdim so result is [N,1]
            torch::Tensor norm_r = r.norm(/*p=*/2, /*dim=*/1, /*keepdim=*/true);   // [N,1]
    
            // 2. gradient factor  g = r / ‖r‖₂  — safe divide
            torch::Tensor norm_safe = norm_r.clone();
            norm_safe.masked_fill_(norm_safe == 0, 1.0);        // avoid 0-div
            torch::Tensor grad_r = r / norm_safe;               // [N,L]
    
            // Zero-out grads where norm==0 to keep derivative well defined
            grad_r.masked_fill_(norm_r == 0, 0.0);
    
            // 3. dual:  Σ_j g_{ij} · d_{ij,k}
            torch::Tensor dual_out =
                (grad_r.unsqueeze(-1) * d)      // [N,L,1] * [N,L,D] = [N,L,D]
                .sum(/*dim=*/1, /*keepdim=*/true);               // → [N,1,D]
    
            return TensorMatDual(norm_r, dual_out);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Construct a TensorMatDual whose real part is `r` and whose dual part
        // is all-zeros with an extra trailing dimension of size `ddim`.
        static TensorMatDual createZero(const torch::Tensor& r, int ddim) {
            // -- validation ---------------------------------------------------------
            if (!r.defined())
                throw std::invalid_argument("createZero(): real tensor `r` is undefined.");
            if (r.dim() != 2)
                throw std::invalid_argument("createZero(): real tensor must be 2-D [N,L].");
            if (ddim <= 0)
                throw std::invalid_argument("createZero(): `ddim` must be positive.");
    
            // -- build zero dual tensor --------------------------------------------
            std::vector<int64_t> dshape = { r.size(0), r.size(1), ddim };  // [N,L,D]
            torch::Tensor d = torch::zeros(dshape,
                                        torch::TensorOptions().dtype(r.dtype())
                                                                .device(r.device()));
    
            return TensorMatDual(r, d);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Produce a TensorMatDual whose real/dual parts are all-zero and
        // share shape, dtype, and device with *this*.
        [[nodiscard]] inline TensorMatDual zeros_like() const {
            if (!r.defined() || !d.defined())
                throw std::runtime_error("TensorMatDual::zeros_like(): tensors are undefined.");
    
            return TensorMatDual(torch::zeros_like(r),   // [N,L]  zeros
                                torch::zeros_like(d));  // [N,L,D] zeros
        }
    
        // ─────────────────────────────────────────────────────────────
        // Deep copy (independent storage for r and d).
        [[nodiscard]] inline TensorMatDual clone() const {
            return TensorMatDual(r.clone(), d.clone());
        }
    
        // ─────────────────────────────────────────────────────────────
        // Concatenate two TensorMatDuals
        //
        //   dim = 0  → stack rows (N–axis)         r, d both cat on dim 0
        //   dim = 1  → stack columns (L–axis)      r, d both cat on dim 1
        //   dim = 2  → stack sensitivities (D–axis) d cat on dim 2, r must coincide
        //
        static TensorMatDual cat(const TensorMatDual &t1,
                                 const TensorMatDual &t2,
                                 int dim = 2)
        {
            // ── basic checks ─────────────────────────────────────────
            if (!t1.r.defined() || !t2.r.defined() || !t1.d.defined() || !t2.d.defined())
                throw std::invalid_argument("TensorMatDual::cat(): undefined tensors.");
            if (t1.r.dim() != 2 || t2.r.dim() != 2 ||
                t1.d.dim() != 3 || t2.d.dim() != 3)
                throw std::invalid_argument("TensorMatDual::cat(): inputs must be [N,L] and [N,L,D].");
            if (dim < 0 || dim > 2)
                throw std::invalid_argument("TensorMatDual::cat(): dim must be 0, 1, or 2.");
    
            // ── concatenate ─────────────────────────────────────────
            torch::Tensor r_cat, d_cat;
    
            if (dim < 2)
            { // rows or columns
                // shapes must match on the untouched axes
                if (t1.r.size(1 - dim) != t2.r.size(1 - dim))
                    throw std::invalid_argument("TensorMatDual::cat(): mismatched shapes for concatenation.");
    
                r_cat = torch::cat({t1.r, t2.r}, dim); // [*,*]
                d_cat = torch::cat({t1.d, t2.d}, dim); // [*,*,D]
            }
            else
            { // dim == 2  (dual axis)
                // real parts must be identical (or at least same shape)
                if (t1.r.sizes() != t2.r.sizes())
                    throw std::invalid_argument("TensorMatDual::cat(): real parts must match when stacking dual axis.");
    
                r_cat = t1.r.clone();                // keep one copy of the matrix
                d_cat = torch::cat({t1.d, t2.d}, 2); // [N,L,D1+D2]
            }
    
            return TensorMatDual(r_cat, d_cat);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Append one column (held in a TensorDual) to a TensorMatDual.
        //   • t1  :  r ∈ ℝ[N,L] , d ∈ ℝ[N,L,D]
        //   • t2  :  r ∈ ℝ[N]   , d ∈ ℝ[N,D]      (or already [N,1] / [N,1,D])
        // Result :  r ∈ ℝ[N,L+1] , d ∈ ℝ[N,L+1,D]
        static TensorMatDual cat(const TensorMatDual& t1, const TensorDual& t2)
        {
            // ── shape checks ─────────────────────────────────────────
            if (!t1.r.defined() || !t1.d.defined() || !t2.r.defined() || !t2.d.defined())
                throw std::invalid_argument("cat(): undefined tensors.");
    
            if (t1.r.dim() != 2 || t1.d.dim() != 3)
                throw std::invalid_argument("cat(): t1 must be [N,L] / [N,L,D].");
    
            // Accept t2 as [N]  or [N,1] for r, and [N,D] or [N,1,D] for d.
            auto rows = t1.r.size(0);
            if (t2.r.dim() == 1) {
                if (t2.r.size(0) != rows)
                    throw std::invalid_argument("cat(): TensorDual length must match number of rows.");
            } else if (t2.r.dim() == 2) {
                if (t2.r.size(0) != rows || t2.r.size(1) != 1)
                    throw std::invalid_argument("cat(): TensorDual matrix must be [N,1].");
            } else {
                throw std::invalid_argument("cat(): TensorDual real part must be 1-D or 2-D.");
            }
    
            if (t2.d.dim() == 2) {
                if (t2.d.size(0) != rows)
                    throw std::invalid_argument("cat(): TensorDual dual part shape mismatch.");
            } else if (t2.d.dim() == 3) {
                if (t2.d.size(0) != rows || t2.d.size(1) != 1)
                    throw std::invalid_argument("cat(): TensorDual dual part must be [N,1,D].");
            } else {
                throw std::invalid_argument("cat(): TensorDual dual part rank must be 2 or 3.");
            }
    
            // ── bring t2 to [N,1]  and [N,1,D]  ----------------------
            torch::Tensor r_vec = (t2.r.dim() == 1) ? t2.r.unsqueeze(1) : t2.r;        // [N,1]
            torch::Tensor d_vec = (t2.d.dim() == 2) ? t2.d.unsqueeze(1) : t2.d;        // [N,1,D]
    
            // ── concatenate along the column axis (dim = 1) ----------
            torch::Tensor r_cat = torch::cat({t1.r, r_vec}, 1);        // [N,L+1]
            torch::Tensor d_cat = torch::cat({t1.d, d_vec}, 1);        // [N,L+1,D]
    
            return TensorMatDual(r_cat, d_cat);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Concatenate a TensorMatDual with a plain tensor:
        //
        //   • dim = 0  → add rows
        //       t2 expected shape: [R_add, L]   or [L]      (broadcast over rows)
        //
        //   • dim = 1  → add columns
        //       t2 expected shape: [C_add]      or [N, C_add]
        //
        // The newly introduced part carries **zero sensitivities** (d = 0).
        static TensorMatDual cat(const TensorMatDual &t1,
                                 const torch::Tensor &t2,
                                 int dim = 1) // default: add columns
        {
            // ── quick checks ─────────────────────────────────────────
            if (!t1.r.defined() || !t1.d.defined() || !t2.defined())
                throw std::invalid_argument("cat(TensorMatDual,Tensor): undefined tensor.");
            if (t1.r.dim() != 2 || t1.d.dim() != 3)
                throw std::invalid_argument("cat(): t1 must be [N,L] / [N,L,D].");
            if (dim != 0 && dim != 1)
                throw std::invalid_argument("cat(): dim must be 0 (rows) or 1 (cols) for plain tensor.");
    
            auto N = t1.r.size(0);
            auto L = t1.r.size(1);
            auto D = t1.d.size(2);
    
            torch::Tensor t2_exp;     // real part to append
            torch::Tensor zeros_dual; // corresponding zero sensitivities
    
            if (dim == 0)
            { //---------------------------------------------------------------- ROWS
                // Accept t2 shapes: [R_add, L]   or [L]
                if (t2.dim() == 1)
                {
                    if (t2.size(0) != L)
                        throw std::invalid_argument("cat(): 1-D t2 length must equal number of columns.");
                    t2_exp = t2.unsqueeze(0); // → [1, L]
                }
                else if (t2.dim() == 2)
                {
                    if (t2.size(1) != L)
                        throw std::invalid_argument("cat(): 2-D t2 must be [..., L].");
                    t2_exp = t2; // [R_add, L]
                }
                else
                {
                    throw std::invalid_argument("cat(): t2 rank must be 1 or 2 for row concat.");
                }
    
                auto R_add = t2_exp.size(0);
                zeros_dual = torch::zeros({R_add, L, D},
                                          t1.d.options()); // [R_add, L, D]
            }
            else
            { //---------------------------------------------------------------- COLUMNS (dim = 1)
                // Accept t2 shapes: [C_add]   or [N, C_add]
                if (t2.dim() == 1)
                {
                    t2_exp = t2.unsqueeze(0).expand({N, -1}); // broadcast row-wise → [N, C_add]
                }
                else if (t2.dim() == 2)
                {
                    if (t2.size(0) != N)
                        throw std::invalid_argument("cat(): 2-D t2 rows must match N.");
                    t2_exp = t2; // [N, C_add]
                }
                else
                {
                    throw std::invalid_argument("cat(): t2 rank must be 1 or 2 for col concat.");
                }
    
                auto C_add = t2_exp.size(1);
                zeros_dual = torch::zeros({N, C_add, D},
                                          t1.d.options()); // [N, C_add, D]
            }
    
            // ── perform concatenation ───────────────────────────────
            torch::Tensor r_cat = torch::cat({t1.r, t2_exp}, dim);     // [N+?, L] or [N, L+?]
            torch::Tensor d_cat = torch::cat({t1.d, zeros_dual}, dim); // [N+?, L, D] or [N, L+?, D]
    
            return TensorMatDual(r_cat, d_cat);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise addition
        [[nodiscard]] inline TensorMatDual operator+(const TensorMatDual& other) const {
            // both parts must be defined and identically shaped
            if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined())
                throw std::invalid_argument("TensorMatDual::operator+: undefined tensors.");
            if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes())
                throw std::invalid_argument("TensorMatDual::operator+: shape mismatch.");
    
            return TensorMatDual(r + other.r, d + other.d);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise addition:  TensorMatDual  +  TensorDual
        //
        //   • t1  : r ∈ ℝ[N,L] , d ∈ ℝ[N,L,D]
        //   • td  : r ∈ ℝ[N]   , d ∈ ℝ[N,D]   (or [N,1] / [N,1,D])
        //
        // The dual’s components are broadcast across the column axis (L).
        // Result preserves layout: r_out ∈ ℝ[N,L] , d_out ∈ ℝ[N,L,D]
        //
        [[nodiscard]] TensorMatDual operator+(const TensorDual& td) const {
            // ── basic validation ─────────────────────────────────────
            if (!r.defined() || !d.defined() || !td.r.defined() || !td.d.defined())
                throw std::invalid_argument("TensorMatDual::operator+: undefined tensors.");
    
            if (r.dim() != 2 || d.dim() != 3)
                throw std::invalid_argument("TensorMatDual::operator+: this must be [N,L]/[N,L,D].");
    
            const auto N = r.size(0);
            const auto L = r.size(1);
            const auto D = d.size(2);
    
            // ── align TensorDual shapes to [N,1] and [N,1,D] ─────────
            torch::Tensor r_vec = (td.r.dim() == 1) ? td.r.unsqueeze(1) : td.r;   // [N,1]
            torch::Tensor d_vec = (td.d.dim() == 2) ? td.d.unsqueeze(1) : td.d;   // [N,1,D]
    
            if (r_vec.size(0) != N || d_vec.size(0) != N || d_vec.size(2) != D)
                throw std::invalid_argument("TensorMatDual::operator+: incompatible TensorDual length or D.");
    
            // ── broadcast-add across columns ─────────────────────────
            torch::Tensor r_sum = r + r_vec.expand({N, L});            // [N,L]
            torch::Tensor d_sum = d + d_vec.expand({N, L, D});         // [N,L,D]
    
            return TensorMatDual(r_sum, d_sum);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Add a scalar to every entry of the real part.
        // Dual sensitivities remain unchanged.
        [[nodiscard]] inline TensorMatDual operator+(double scalar) const {
            if (!r.defined())
                throw std::runtime_error("TensorMatDual::operator+(scalar): real tensor is undefined.");
    
            // `d` is left as-is because ∂/∂x (x + c) = 1 ⇒ dual stays unchanged.
            return TensorMatDual(r + scalar, d);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise subtraction
        [[nodiscard]] inline TensorMatDual operator-(const TensorMatDual& other) const {
            if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined())
                throw std::invalid_argument("TensorMatDual::operator-: undefined tensors.");
            if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes())
                throw std::invalid_argument("TensorMatDual::operator-: shape mismatch.");
    
            return TensorMatDual(r - other.r, d - other.d);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Subtract a scalar from every entry of the real part.
        // Dual sensitivities remain unchanged.
        [[nodiscard]] inline TensorMatDual operator-(double scalar) const {
            if (!r.defined())
                throw std::runtime_error("TensorMatDual::operator-(scalar): real tensor is undefined.");
    
            return TensorMatDual(r - scalar, d);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise equality test on the **real** parts.
        // (Dual parts are intentionally ignored.)
        [[nodiscard]] inline torch::Tensor operator==(const TensorMatDual& other) const {
            if (!r.defined() || !other.r.defined())
                throw std::invalid_argument("TensorMatDual::operator==: real tensors are undefined.");
            if (r.sizes() != other.r.sizes())
                throw std::invalid_argument("TensorMatDual::operator==: shape mismatch for comparison.");
    
            return r == other.r;        // bool tensor, same [N,L] shape
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Unary negation:  -(TensorMatDual)
        [[nodiscard]] inline TensorMatDual operator-() const {
            return TensorMatDual(-r, -d);   // element-wise negation of both parts
        }
    
        // ─────────────────────────────────────────────────────────────
        // Scalar multiplication:  TensorMatDual * c
        [[nodiscard]] inline TensorMatDual operator*(double scalar) const {
            return TensorMatDual(r * scalar,        // scale real part
                                d * scalar);       // scale dual part (linear rule)
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise division of two TensorMatDuals
        //   result.r = r / s
        //   result.d = (d * s - r * s') / s²
        [[nodiscard]] inline TensorMatDual operator/(const TensorMatDual& other) const {
            // shape / definition checks
            if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined())
                throw std::invalid_argument("TensorMatDual::operator/: undefined tensors.");
            if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes())
                throw std::invalid_argument("TensorMatDual::operator/: shape mismatch.");
    
            // guard against divide-by-zero in the real part of the denominator
            if (torch::any(other.r == 0).item<bool>())
                throw std::runtime_error("TensorMatDual::operator/: division by zero in denominator.");
    
            // shorthand
            const torch::Tensor& f  =  r;           // numerator  (this)
            const torch::Tensor& fp = d;            // its dual   [N,L,D]
            const torch::Tensor& g  =  other.r;     // denominator
            const torch::Tensor& gp = other.d;      // its dual   [N,L,D]
    
            // real part
            torch::Tensor r_out = f / g;            // [N,L]
    
            // dual part:  (fp * g - f * gp) / g²
            torch::Tensor g2   = g.square();                     // [N,L]
            torch::Tensor d_out =
                (fp * g.unsqueeze(-1) - f.unsqueeze(-1) * gp) /  // broadcast on D
                g2.unsqueeze(-1);                                // [N,L,1]
    
            return TensorMatDual(r_out, d_out);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise division by a plain tensor (scalar, vector, or
        // matrix broadcastable to [N,L]).
        //
        //   r_out = r / denom
        //   d_out = d / denom          (broadcast along D)
        //
        // Throws if `denom` is undefined or contains zeros.
        [[nodiscard]] inline TensorMatDual operator/(const torch::Tensor& denom) const {
            if (!denom.defined())
                throw std::invalid_argument("TensorMatDual::operator/: denominator tensor is undefined.");
    
            // guard against divide-by-zero
            if (torch::any(denom == 0).item<bool>())
                throw std::runtime_error("TensorMatDual::operator/: division by zero in denominator tensor.");
    
            // PyTorch will broadcast `denom` to match [N,L]; we add a trailing
            // singleton axis so it also broadcasts across D for the dual part.
            torch::Tensor r_out = r / denom;                   // [N,L]  ÷ broadcast
            torch::Tensor d_out = d / denom.unsqueeze(-1);     // [N,L,D] ÷ [*,*,1]
    
            return TensorMatDual(r_out, d_out);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Divide by a scalar:  TensorMatDual / c
        [[nodiscard]] inline TensorMatDual operator/(double scalar) const {
            if (scalar == 0.0)
                throw std::runtime_error("TensorMatDual::operator/(scalar): division by zero.");
    
            return TensorMatDual(r / scalar,          // real part
                                d / scalar);         // dual part
        }
    
        
        // ─────────────────────────────────────────────────────────────
        // Fancy-index the matrix and its sensitivities.
        //
        //  • `indices` may specify up to two axes (rows, columns).
        //    Anything on the dual axis (D) is inferred as ":".
        //
        //  • After indexing we re-add singleton axes so the result
        //    remains [N,L]  for r   and  [N,L,D]  for d.
        [[nodiscard]]
        TensorMatDual index(const std::vector<torch::indexing::TensorIndex>& indices) const
        {
            if (!r.defined() || !d.defined())
                throw std::invalid_argument("TensorMatDual::index(): undefined tensors.");
    
            // ── real part ────────────────────────────────────────────
            torch::Tensor r_sub = r.index(indices);          // rank ≤ 2
    
            // ── dual part (need ':' on D-axis) ──────────────────────
            std::vector<torch::indexing::TensorIndex> idx_d = indices;
            if (idx_d.size() < 3) idx_d.push_back(torch::indexing::Slice());  // keep all D
            torch::Tensor d_sub = d.index(idx_d);            // rank ≤ 3
    
            // ── restore invariant ranks ─────────────────────────────
            while (r_sub.dim() < 2) r_sub = r_sub.unsqueeze(-1);             // [N] → [N,1]
            while (d_sub.dim() < 3) d_sub = d_sub.unsqueeze(-2);             // [N,D] → [N,1,D]
    
            return TensorMatDual(r_sub, d_sub);
        }
    
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Extract a single row (first axis) without collapsing rank.
        //
        [[nodiscard]] inline TensorMatDual index(int idx) const {
            if (!r.defined() || !d.defined())
                throw std::invalid_argument("TensorMatDual::index(int): undefined tensors.");
    
            if (idx < 0 || idx >= r.size(0))
                throw std::out_of_range("TensorMatDual::index(int): row index out of range.");
    
            // Slice row `idx`, then unsqueeze so shape stays [1,L] / [1,L,D]
            torch::Tensor r_row = r.index({ idx }).unsqueeze(0);   // [1, L]
            torch::Tensor d_row = d.index({ idx }).unsqueeze(0);   // [1, L, D]
    
            return TensorMatDual(r_row, d_row);
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // Boolean-mask index on the first (row) axis.
        // `mask` must be 1-D bool with length N or broadcastable to it.
        [[nodiscard]]
        TensorMatDual index(const torch::Tensor& mask) const {
            if (!mask.defined())
                throw std::invalid_argument("TensorMatDual::index(mask): mask is undefined.");
            if (mask.dtype() != torch::kBool)
                throw std::invalid_argument("TensorMatDual::index(mask): mask must be bool.");
            if (mask.dim() != 1 || mask.size(0) != r.size(0))
                throw std::invalid_argument("TensorMatDual::index(mask): mask must be 1-D of length N.");
    
            using torch::indexing::Slice;
    
            // Real part: pick masked rows
            torch::Tensor r_sel = r.index({ mask, Slice() });                  // [M,L]
    
            // Dual part: masked rows, all cols, all D
            torch::Tensor d_sel = d.index({ mask, Slice(), Slice() });         // [M,L,D]
    
            return TensorMatDual(r_sel, d_sel);
        }
    
    
    
    
    
    
    
        // ─────────────────────────────────────────────────────────────
        // Element-wise absolute value with first-order sensitivities.
        //
        //   r_out = |r|
        //   d_out = sgn(r) · d
        //   where sgn(z) = z / |z|  for complex (matches PyTorch -> real part’s sign)
        //
        // If r is complex, PyTorch’s `torch::abs` gives the magnitude and
        // `torch::sgn` returns z / |z|, which has the correct derivative.
        //
        [[nodiscard]] inline TensorMatDual abs() const {
            if (!r.defined() || !d.defined())
                throw std::invalid_argument("TensorMatDual::abs(): undefined tensors.");
    
            // 1. absolute value of the real part
            torch::Tensor r_abs = torch::abs(r);            // [N,L]
    
            // 2. sign factor  (+1, -1, or complex phase)
            torch::Tensor sign_r = torch::sgn(r);           // [N,L]
    
            // 3. propagate to dual part (broadcast along D)
            torch::Tensor d_abs = sign_r.unsqueeze(-1) * d; // [N,L,1] * [N,L,D] → [N,L,D]
    
            return TensorMatDual(r_abs, d_abs);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Limited-arity einsum:  (TensorMatDual , TensorDual) → TensorDual
        // Supports two operands only; ‘z’ is the reserved dual axis label.
        static TensorDual einsum(const std::string &eq,
                                 const TensorMatDual &A,
                                 const TensorDual &b)
        {
            if (eq.find('z') != std::string::npos)
                throw std::invalid_argument("einsum spec must not already contain the reserved axis 'z'.");

            if (b.r.device() != A.r.device() || b.r.dtype() != A.r.dtype())
                throw std::invalid_argument("dtype / device mismatch between operands.");

            // real contraction
            torch::Tensor r_out = torch::einsum(eq, {A.r, b.r});
    
            // split equation  "subA,subB->subOut"
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            // derivative wrt A :  A.d carries extra 'z'
            std::string eq_dA = subA + "z," + subB + "->" + subO + "z";
            torch::Tensor d1 = torch::einsum(eq_dA, {A.d, b.r});
    
            // derivative wrt b :  b.d carries extra 'z'
            std::string eq_dB = subA + "," + subB + "z->" + subO + "z";
            torch::Tensor d2 = torch::einsum(eq_dB, {A.r, b.d});
    
            return TensorDual(std::move(r_out), std::move(d1 + d2));
        }


        // ─────────────────────────────────────────────────────────────
        // Limited-arity einsum:  (TensorMatDual , TensorDual) → TensorDual
        // Supports two operands only; ‘z’ is the reserved dual axis label.
        static TensorDual einsum(const std::string &eq,
                                 const TensorMatDual &A,
                                 const torch::Tensor &b)
        {
            if (eq.find('z') != std::string::npos)
                throw std::invalid_argument("einsum spec must not already contain the reserved axis 'z'.");

            if (b.device() != A.r.device() || b.dtype() != A.r.dtype())
                throw std::invalid_argument("dtype / device mismatch between operands.");

            // real contraction
            torch::Tensor r_out = torch::einsum(eq, {A.r, b});
    
            // split equation  "subA,subB->subOut"
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            // derivative wrt A :  A.d carries extra 'z'
            std::string eq_dA = subA + "z," + subB + "->" + subO + "z";
            torch::Tensor d1 = torch::einsum(eq_dA, {A.d, b});
    
    
            return TensorDual(std::move(r_out), std::move(d1));
        }


        // ─────────────────────────────────────────────────────────────
        // Limited-arity einsum:  (TensorMatDual , TensorDual) → TensorDual
        // Supports two operands only; ‘z’ is the reserved dual axis label.
        static TensorDual einsum(const std::string &eq,
                                 const torch::Tensor &b,
                                 const TensorMatDual &A
                                 )
        {
            if (eq.find('z') != std::string::npos)
                throw std::invalid_argument("einsum spec must not already contain the reserved axis 'z'.");

            if (b.device() != A.r.device() || b.dtype() != A.r.dtype())
                throw std::invalid_argument("dtype / device mismatch between operands.");

            // real contraction
            torch::Tensor r_out = torch::einsum(eq, {b, A.r});
    
            // split equation  "subA,subB->subOut"
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            std::string subB = eq.substr(0, comma);
            std::string subA = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            // derivative wrt A :  A.d carries extra 'z'
            std::string eq_dA = subB+","+subA + "z->" + subO + "z";
            torch::Tensor d1 = torch::einsum(eq_dA, {b, A.d});
    
    
            return TensorDual(std::move(r_out), std::move(d1));
        }

    
        // ─────────────────────────────────────────────────────────────
        // einsum( TensorDual , TensorMatDual )  →  TensorDual
        // Two-argument variant; ‘z’ tags the dual axis on the operand
        // that carries sensitivities (here: both operands).
        static TensorDual einsum(const std::string &eq,
                                 const TensorDual &a,
                                 const TensorMatDual &B)
        {
            // ── basic validation ────────────────────────────────────
            if (!a.r.defined() || !a.d.defined() ||
                !B.r.defined() || !B.d.defined())
                throw std::invalid_argument("einsum: undefined tensors.");
    
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            // split equation
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            // ── real contraction ────────────────────────────────────
            torch::Tensor r_out = torch::einsum(eq, {a.r, B.r});
    
            // ── dual terms (product rule) ───────────────────────────
            std::string eq_dA = subA + "z," + subB + "->" + subO + "z";
            std::string eq_dB = subA + "," + subB + "z->" + subO + "z";
    
            torch::Tensor d1 = torch::einsum(eq_dA, {a.d, B.r});
            torch::Tensor d2 = torch::einsum(eq_dB, {a.r, B.d});
    
            return TensorDual(std::move(r_out), std::move(d1 + d2));
        }
    
    
        // ─────────────────────────────────────────────────────────────
        // einsum( TensorMatDual , TensorMatDual )  →  TensorMatDual
        // Two-argument variant; both operands carry sensitivities
        // along their trailing ‘z’ axis (size D).
        static TensorMatDual einsum(const std::string &eq,
                                    const TensorMatDual &A,
                                    const TensorMatDual &B)
        {
            // ── basic validation ────────────────────────────────────
            auto comma = eq.find(',');
            auto arrow = eq.find("->");
            if (comma == std::string::npos || arrow == std::string::npos)
                throw std::invalid_argument("einsum string must contain ',' and '->'.");
    
            if (!A.r.defined() || !A.d.defined() ||
                !B.r.defined() || !B.d.defined())
                throw std::invalid_argument("einsum: undefined tensors.");
    
            // split equation  "subA,subB->subO"
            std::string subA = eq.substr(0, comma);
            std::string subB = eq.substr(comma + 1, arrow - comma - 1);
            std::string subO = eq.substr(arrow + 2);
    
            // ── real contraction ────────────────────────────────────
            torch::Tensor r_out = torch::einsum(eq, {A.r, B.r});
    
            // ── dual contractions (product rule) ────────────────────
            // term 1:  A.d carries ‘z’
            std::string eq_dA = subA + "z," + subB + " -> " + subO + "z";
            torch::Tensor d1 = torch::einsum(eq_dA, {A.d, B.r});
    
            // term 2:  B.d carries ‘z’
            std::string eq_dB = subA + "," + subB + "z -> " + subO + "z";
            torch::Tensor d2 = torch::einsum(eq_dB, {A.r, B.d});
    
            // combined dual
            torch::Tensor d_out = d1 + d2; // [*,*,D] (trailing z-axis)
    
            return TensorMatDual(std::move(r_out), std::move(d_out));
        }
    
        // ─────────────────────────────────────────────────────────────
        // Row- or column-wise max with dual propagation
        //
        //   • dim = 0  → max over rows, keepdim=true  → r_out [1,L]
        //   • dim = 1  → max over cols, keepdim=true  → r_out [N,1]
        //
        //   The corresponding dual entries are gathered from `d`.
        //
        [[nodiscard]] TensorMatDual max(int dim = 1) const {
            // ── checks ───────────────────────────────────────────────
            if (!r.defined() || !d.defined())
                throw std::invalid_argument("TensorMatDual::max(): tensors undefined.");
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("TensorMatDual::max(): dim must be 0 or 1.");
    
            // ── compute max and indices on the real part ─────────────
            auto max_pair = r.is_complex()
                            ? torch::max(torch::real(r), /*dim=*/dim, /*keepdim=*/true)
                            : torch::max(r,              /*dim=*/dim, /*keepdim=*/true);
    
            torch::Tensor r_max = std::get<0>(max_pair);   // [N,1] or [1,L]
            torch::Tensor idx   = std::get<1>(max_pair);   // same shape, long dtype
    
            // ── gather corresponding dual entries ───────────────────
            // Expand `idx` to have a trailing axis so it matches d’s rank.
            //   For dim==0: idx shape [1,L]  → [1,L,1]  (unsq at -1)
            //   For dim==1: idx shape [N,1]  → [N,1,1]
            torch::Tensor idx_expanded = idx.unsqueeze(-1).expand({-1, -1, d.size(2)});
    
            torch::Tensor d_max = torch::gather(d, /*dim=*/dim, idx_expanded); // [N,1,D] or [1,L,D]
    
            return TensorMatDual(r_max, d_max);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Row- or column-wise minimum with dual propagation
        //
        //   • dim = 0  → min over rows  → r_out [1,L] , d_out [1,L,D]
        //   • dim = 1  → min over cols  → r_out [N,1] , d_out [N,1,D]
        //
        [[nodiscard]] TensorMatDual min(int dim = 1) const {
            if (!r.defined() || !d.defined())
                throw std::invalid_argument("TensorMatDual::min(): tensors undefined.");
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("TensorMatDual::min(): dim must be 0 or 1.");
    
            // real part: complex → compare real component (change to abs() if preferred)
            auto min_pair = r.is_complex()
                            ? torch::min(torch::real(r), dim, /*keepdim=*/true)
                            : torch::min(r,              dim, /*keepdim=*/true);
    
            torch::Tensor r_min = std::get<0>(min_pair);     // [N,1] or [1,L]
            torch::Tensor idx   = std::get<1>(min_pair);     // same shape
    
            // broadcast indices to gather the dual entries
            torch::Tensor idx_exp = idx.unsqueeze(-1).expand({-1, -1, d.size(2)});
            torch::Tensor d_min   = torch::gather(d, dim, idx_exp);  // [N,1,D] or [1,L,D]
    
            return TensorMatDual(r_min, d_min);
        }
    
        // ─────────────────────────────────────────────────────────────
        // In-place masked assignment (TensorMatDual ← value at mask)
        //
        //   • mask  : bool tensor, same shape as r   ([N,L])
        //   • value : TensorMatDual with identical shapes
        //
        inline void index_put_(const torch::Tensor &mask,
                               const TensorMatDual &value)
        {
            // ── validation ──────────────────────────────────────────
            if (!mask.defined() || mask.dtype() != torch::kBool)
                throw std::invalid_argument("index_put_: mask must be a defined bool tensor.");
            if (!value.r.defined() || !value.d.defined())
                throw std::invalid_argument("index_put_: value tensors undefined.");
    
            if (mask.sizes() != r.sizes())
                throw std::invalid_argument("index_put_: mask shape must match real tensor shape.");
            if (value.r.sizes() != r.sizes() || value.d.sizes() != d.sizes())
                throw std::invalid_argument("index_put_: value shape mismatch.");
    
            // ── real part update ───────────────────────────────────
            r.index_put_({mask}, value.r);
    
            // ── dual part update  (broadcast mask along D) ─────────
            auto mask_d = mask.unsqueeze(-1).expand_as(d); // [N,L,D]
            d.index_put_({mask_d}, value.d);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Masked scalar assign:  this(mask) = value   (dual reset to 0)
        //
        // `mask` may specify up to the two matrix axes (rows / cols).
        // The dual axis (D) is automatically kept unchanged.
        inline void index_put_(const std::vector<torch::indexing::TensorIndex> &mask,
                               double value)
        {
            using torch::indexing::Slice;
    
            // shape sanity-check
            if (mask.size() > 2)
                throw std::invalid_argument("index_put_: mask rank exceeds [N,L] layout.");
    
            // ── update real part ────────────────────────────────────
            r.index_put_(mask, value);
    
            // ── update dual part – append ':' for the D-axis ───────
            std::vector<torch::indexing::TensorIndex> mask_d = mask;
            mask_d.push_back(Slice()); // keep all sensitivities (D)
    
            d.index_put_(mask_d, 0.0); // zero-out corresponding dual slots
        }
    
        // ─────────────────────────────────────────────────────────────
        // In-place masked assignment (TensorMatDual ← TensorDual)
        //
        //   • `mask` selects a sub-matrix of `r` (rows/cols); rank ≤ 2.
        //   • `dim` tells us whether `value` should be broadcast down the
        //     row-axis (`dim = 0`) or column-axis (`dim = 1`).
        //   • The dual part of `value` is expanded so it has a trailing
        //     sensitivity axis and a singleton on the broadcast dimension.
        //
        void index_put_(const std::vector<torch::indexing::TensorIndex> &mask,
                        const TensorDual &value,
                        int dim = 1) // 0 = rows, 1 = cols
        {
            using torch::indexing::Slice;
    
            // ── validation ──────────────────────────────────────────
            if (!value.r.defined() || !value.d.defined())
                throw std::invalid_argument("index_put_: value tensors undefined.");
            if (mask.size() > 2)
                throw std::invalid_argument("index_put_: mask rank exceeds [N,L] layout.");
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("index_put_: dim must be 0 (rows) or 1 (cols).");
    
            // ── bring value to the right rank ──────────────────────
            // row-vector or column-vector broadcast depending on `dim`
            torch::Tensor r_val, d_val;
    
            if (dim == 0)
            {                                                                  // broadcast across rows → shape [1,L] / [1,L,D]
                r_val = (value.r.dim() == 1) ? value.r.unsqueeze(0) : value.r; // [1,L]
                d_val = (value.d.dim() == 2) ? value.d.unsqueeze(0) : value.d; // [1,L,D]
            }
            else
            {                                                                  // broadcast across cols → shape [N,1] / [N,1,D]
                r_val = (value.r.dim() == 1) ? value.r.unsqueeze(1) : value.r; // [N,1]
                d_val = (value.d.dim() == 2) ? value.d.unsqueeze(1) : value.d; // [N,1,D]
            }
    
            // ── assign real part ───────────────────────────────────
            r.index_put_(mask, r_val);
    
            // ── assign dual part (append ':' for the D-axis) ──────
            auto mask_d = mask;        // copy ≈ [rows, cols]
            mask_d.push_back(Slice()); // keep all sensitivities
            d.index_put_(mask_d, d_val);
        }
    
        // ─────────────────────────────────────────────────────────────
        // In-place masked assign: this(mask) = value  (TensorMatDual ← TensorMatDual)
        //
        // `mask` selects rows/cols (rank ≤ 2).  The dual axis (D) is
        // appended automatically so the full sensitivity stack is updated.
        inline void index_put_(const std::vector<torch::indexing::TensorIndex> &mask,
                               const TensorMatDual &value)
        {
            using torch::indexing::Slice;
    
            // ── sanity checks ───────────────────────────────────────
            if (!value.r.defined() || !value.d.defined())
                throw std::invalid_argument("index_put_: value tensors undefined.");
            if (mask.size() > 2)
                throw std::invalid_argument("index_put_: mask rank exceeds [N,L] layout.");
            if (value.r.sizes() != r.sizes() || value.d.sizes() != d.sizes())
                throw std::invalid_argument("index_put_: value shape mismatch.");
    
            // ── real part update ───────────────────────────────────
            r.index_put_(mask, value.r);
    
            // ── dual part update (append ':' to cover D axis) ──────
            auto mask_d = mask;        // copy row/col selectors
            mask_d.push_back(Slice()); // keep all D
            d.index_put_(mask_d, value.d);
        }
    
        // ─────────────────────────────────────────────────────────────
        // In-place masked assignment with a plain tensor.
        //   • real entries at `mask` are replaced by `value`
        //   • corresponding dual entries are zeroed
        inline void index_put_(const std::vector<torch::indexing::TensorIndex> &mask,
                               const torch::Tensor &value)
        {
            using torch::indexing::Slice;
    
            // ── validation ──────────────────────────────────────────
            if (!value.defined())
                throw std::invalid_argument("index_put_: value tensor is undefined.");
            if (mask.size() > 2)
                throw std::invalid_argument("index_put_: mask rank exceeds [N,L] layout.");
    
            torch::Tensor r_slice = r.index(mask); // shape selected by mask
            if (value.sizes() != r_slice.sizes())
                throw std::invalid_argument("index_put_: value shape incompatible with mask.");
    
            // ── update real part ───────────────────────────────────
            r.index_put_(mask, value);
    
            // ── zero-out matching dual part ────────────────────────
            std::vector<int64_t> zero_shape = r_slice.sizes().vec(); // [*,*]
            zero_shape.push_back(d.size(2));                         // add D axis → [*,*,D]
            torch::Tensor zeros = torch::zeros(zero_shape, d.options());
    
            auto mask_d = mask;        // copy row/col selectors
            mask_d.push_back(Slice()); // ':' on D axis
            d.index_put_(mask_d, zeros);
        }
    
        // ─────────────────────────────────────────────────────────────
        // Add a singleton matrix axis to a TensorDual and return a TensorMatDual.
        //   • dim = 0  → r: [1,L] , d: [1,L,D]   (broadcast down rows later)
        //   • dim = 1  → r: [N,1] , d: [N,1,D]   (broadcast across cols later)
        static TensorMatDual unsqueeze(const TensorDual& x, int dim = 1) {
            if (!x.r.defined() || !x.d.defined())
                throw std::invalid_argument("unsqueeze(): TensorDual tensors are undefined.");
            if (dim < 0 || dim > 1)
                throw std::invalid_argument("unsqueeze(): dim must be 0 (row) or 1 (col).");
    
            // real part gets a new singleton axis
            torch::Tensor r_mat = x.r.unsqueeze(dim);         // [1,L] or [N,1]
    
            // dual part gets the same singleton, becoming 3-D
            torch::Tensor d_mat = x.d.unsqueeze(dim);         // [1,L,D] or [N,1,D]
    
            return TensorMatDual(r_mat, d_mat);
        }
    
    };
}