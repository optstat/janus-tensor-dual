#pragma once
#include <torch/torch.h>
#include <torch/index.h>
#include <type_traits> // For std::is_scalar
#include <vector>

/**
 * @brief TensorHyperDual stores first‑ and second‑order sensitivities w.r.t. a size‑N state vector.
 *
 *  Supports **real _or_ complex** scalar types (Float, Double, ComplexFloat, ComplexDouble).
 *
 *  Shapes
 *  ───────
 *    r : [N]          – value (state vector)
 *    d : [N, D]       – first‑order sensitivities
 *    h : [N, D, D]    – second‑order sensitivities (symmetric but stored dense)
 *
 *  The class follows PyTorch’s memory‑management semantics and allows seamless CPU/GPU moves.
 */
class TensorHyperDual {
    public:
        /*──────────────────────────── Data members ─────────────────────────────*/
        torch::Tensor r; ///< primal values
        torch::Tensor d; ///< dual (∂x / ∂ϵₖ)
        torch::Tensor h; ///< hyper‑dual (∂²x / ∂ϵₖ∂ϵⱼ)
    
        torch::Dtype  dtype_  = torch::kFloat64; ///< scalar type
        torch::Device device_ = torch::kCPU;     ///< storage device
    
        /*──────────────────────────── Constructors ─────────────────────────────*/
    
        /**
         * @param N      state dimension
         * @param D      dual dimension (number of perturbation directions)
         * @param dtype  scalar dtype (real or complex)
         * @param device torch::kCPU / torch::kCUDA
         */
        explicit TensorHyperDual(int N             = 1,
                                 int D             = 1,
                                 torch::Dtype dtype   = torch::kFloat64,
                                 torch::Device device = torch::kCPU)
            : r(torch::zeros({N},       torch::TensorOptions().dtype(dtype).device(device))),
              d(torch::zeros({N, D},    torch::TensorOptions().dtype(dtype).device(device))),
              h(torch::zeros({N, D, D}, torch::TensorOptions().dtype(dtype).device(device))),
              dtype_(dtype),
              device_(device) {}
    
        /// Construct from existing tensors (shape / dtype / device sanity checks)
        TensorHyperDual(const torch::Tensor& r_,
                        const torch::Tensor& d_,
                        const torch::Tensor& h_) {
            validateTensors(r_, d_, h_);
            r = r_;
            d = d_;
            h = h_;
            dtype_  = torch::typeMetaToScalarType(r.dtype());
            device_ = r.device();
        }
    
        /*──────────────────────────── Factory helpers ──────────────────────────*/
        static TensorHyperDual zeros(int N, int D,
                                     torch::Dtype dtype   = torch::kFloat64,
                                     torch::Device device = torch::kCPU) {
            return TensorHyperDual(N, D, dtype, device);
        }
        /**
         * @brief Promote a TensorDual to TensorHyperDual.
         *
         *  •  r and d are copied by reference (cheap, shared storage).
         *  •  h is freshly allocated and filled with zeros on the same device/dtype.
         *
         *  Shapes
         *    x.r : [N]
         *    x.d : [N, D]
         *    h   : [N, D, D]   (new – no batch axis)
         */
        explicit TensorHyperDual(const TensorDual& x)
            : r(x.r),
            d(x.d),
            h(torch::zeros({x.d.size(0),          // N
                            x.d.size(1),          // D
                            x.d.size(1)},         // D
                            x.d.options())),       // preserves dtype & device
            dtype_(x.r.dtype().toScalarType()),
            device_(x.r.device()) {
            validateTensors(r, d, h);               // sanity check
        }    

        /*──────────────────────────── Device / dtype moves ─────────────────────*/
        /** Move *in‑place* to a new device. */
        void to(torch::Device new_device) {
            r = r.to(new_device);
            d = d.to(new_device);
            h = h.to(new_device);
            device_ = new_device;
        }
    
        /** Cast *in‑place* to a new scalar type (e.g. Float → ComplexFloat). */
        void to(torch::Dtype new_dtype) {
            r = r.to(new_dtype);
            d = d.to(new_dtype);
            h = h.to(new_dtype);
            dtype_ = new_dtype;
        }
    
        /*──────────────────────────── Complex utilities ────────────────────────*/
        /** Return the complex conjugate (functional, non‑mutating). */
        [[nodiscard]] TensorHyperDual conj() const {
            return TensorHyperDual(r.conj(), d.conj(), h.conj());
        }
    
        /** In‑place conjugation – cheaper when temporaries matter. */
        void conj_() {
            r.conj_();
            d.conj_();
            h.conj_();
        }
    
    private:
        /*──────────────────────────── Validation helper ────────────────────────*/
        static void validateTensors(const torch::Tensor& r_,
                                    const torch::Tensor& d_,
                                    const torch::Tensor& h_) {
            TORCH_CHECK(r_.dim() == 1, "TensorHyperDual: r must be rank‑1 [N]");
            TORCH_CHECK(d_.dim() == 2, "TensorHyperDual: d must be rank‑2 [N,D]");
            TORCH_CHECK(h_.dim() == 3, "TensorHyperDual: h must be rank‑3 [N,D,D]");
            TORCH_CHECK(r_.size(0) == d_.size(0) && r_.size(0) == h_.size(0),
                        "TensorHyperDual: mismatch in state dimension N");
            TORCH_CHECK(d_.size(1) == h_.size(1) && d_.size(1) == h_.size(2),
                        "TensorHyperDual: mismatch in dual dimension D");
            TORCH_CHECK(r_.dtype() == d_.dtype() && r_.dtype() == h_.dtype(),
                        "TensorHyperDual: tensors must share dtype");
            TORCH_CHECK(r_.device() == d_.device() && r_.device() == h_.device(),
                        "TensorHyperDual: tensors must share device");
        }





public:
    TensorHyperDual clone() const {
        return TensorHyperDual(r.clone(), d.clone(), h.clone());
    }



    /**
     * @brief Return a new TensorHyperDual whose r, d, h blocks are stored contiguously.
     *
     * Handy after slicing or transposing, when any of the underlying tensors
     * may have become non-contiguous.
     */
    TensorHyperDual contiguous() const {
        return TensorHyperDual(r.contiguous(),
                            d.contiguous(),
                            h.contiguous());
    }

    TensorMatHyperDual eye();

    /**
     * @brief Stream-insertion operator for TensorHyperDual.
     *
     * Prints the dtype, device, and the r / d / h blocks.
     */
    friend std::ostream& operator<<(std::ostream& os, const TensorHyperDual& obj) {
        using c10::toString;          // helper to stringify dtype

        os << "TensorHyperDual(dtype=" << toString(obj.r.dtype())
        << ", device=" << obj.r.device() << ") {\n"
        << "  r: " << obj.r << '\n'
        << "  d: " << obj.d << '\n'
        << "  h: " << obj.h << '\n'
        << '}';

        return os;
    }

    /**
     * @brief Sum the TensorHyperDual along a given dimension.
     *
     * @param dim  Dimension index to reduce:
     *               0 → over the state dimension N
     *               1 → over the dual dimension D
     *               2 → only valid for h (second D axis)
     *
     * @param keepdim  If true, keeps a singleton dimension (PyTorch style).
     *
     * @return A new TensorHyperDual with each block summed along @p dim.
     */
    TensorHyperDual sum(int64_t dim = 0, bool keepdim = false) const
    {
        torch::Tensor new_r, new_d, new_h;

        switch (dim) {
        case 0:        // sum over N
            new_r = r.sum(/*dim=*/0, /*keepdim=*/keepdim);
            new_d = d.sum(/*dim=*/0, /*keepdim=*/keepdim);
            new_h = h.sum(/*dim=*/0, /*keepdim=*/keepdim);
            break;

        case 1:        // sum over first D axis
            // r has no axis-1, so leave it untouched (or broadcast & sum to match?)
            new_r = r.clone();                     // or throw if that’s undesired
            new_d = d.sum(/*dim=*/1, keepdim);
            new_h = h.sum(/*dim=*/1, keepdim);
            break;

        case 2:        // sum over second D axis (only h has it)
            new_r = r.clone();
            new_d = d.clone();
            new_h = h.sum(/*dim=*/2, keepdim);
            break;

        default:
            TORCH_CHECK(false, "TensorHyperDual::sum – dim must be 0, 1, or 2.");
        }

        return TensorHyperDual(new_r, new_d, new_h);
    }

    
    /**
     * @brief Element-wise square of a TensorHyperDual.
     *
     * r_new = r²
     * d_new = 2 r d
     * h_new = 2 d ⊗ d  +  2 r h
     */
    TensorHyperDual square() const
    {
        // r²  ────────────────────────────────────────────────
        auto rsq = r * r;                                    // [N]

        // 2 r d  ─────────────────────────────────────────────
        // r.unsqueeze(1) → [N,1] broadcasts against d [N,D]
        auto dn  = 2.0 * r.unsqueeze(1) * d;                 // [N,D]

        // 2 d⊗d  ─────────────────────────────────────────────
        // [N,D,1] * [N,1,D] → broadcast outer product [N,D,D]
        auto term1 = 2.0 * d.unsqueeze(2) * d.unsqueeze(1);  // [N,D,D]

        // 2 r h  ─────────────────────────────────────────────
        auto term2 = 2.0 * r.unsqueeze(1).unsqueeze(2) * h;  // [N,D,D]

        auto hn = term1 + term2;                             // [N,D,D]

        return TensorHyperDual(rsq, dn, hn);
    }
    

    /**
     * @brief Element-wise square-root of a TensorHyperDual.
     *
     * r_new = √r
     * d_new = d / (2 √r)
     * h_new = − d⊗d / (4 r^{3/2}) + h / (2 √r)
     *
     * If any element of `r` is negative and the tensor is real-typed, the inputs
     * are promoted to complex so the result is mathematically well-defined.
     */
    TensorHyperDual sqrt() const
    {
        // ─── Promote to complex if needed ───────────────────────────────────────
        torch::Tensor rc = r, dc = d, hc = h;               // copies of handles

        if (!r.is_complex() && (r < 0).any().item<bool>()) {
            rc = torch::complex(r, torch::zeros_like(r));
            dc = torch::complex(d, torch::zeros_like(d));
            hc = torch::complex(h, torch::zeros_like(h));
        }

        // ─── √r  (broadcast helpers) ────────────────────────────────────────────
        auto root  = rc.sqrt();                             // [N]
        auto root2 = 2.0 * root;                            // [N]
        auto root32 = 4.0 * root.pow(3);                    // 4 r^{3/2}  [N]

        // ─── d_new = d / (2 √r) ────────────────────────────────────────────────
        auto dn = dc / root2.unsqueeze(1);                  // [N,D]

        // ─── h_new = − d⊗d / (4 r^{3/2})  +  h / (2 √r) ───────────────────────
        auto term1 = -(dc.unsqueeze(2) * dc.unsqueeze(1))   // d⊗d, shape [N,D,D]
                    / root32.unsqueeze(1).unsqueeze(2);

        auto term2 = hc / root2.unsqueeze(1).unsqueeze(2);  // h / (2√r)

        auto hn = term1 + term2;                            // [N,D,D]

        return TensorHyperDual(root, dn, hn);
    }

    
    /**
     * @brief Element-wise addition of two TensorHyperDual objects.
     *
     * All three blocks (r, d, h) must share dtype, device, and compatible shapes.
     */
    TensorHyperDual operator+(const TensorHyperDual& other) const {
        TORCH_CHECK(r.dtype()   == other.r.dtype()   &&
                    r.device()  == other.r.device(),
                    "TensorHyperDual::operator+ – dtype or device mismatch");

        return TensorHyperDual(r + other.r,
                            d + other.d,
                            h + other.h);
    }
    
    /**
     * @brief Unary minus: returns a TensorHyperDual whose components are negated.
     */
    TensorHyperDual operator-() const {
        return TensorHyperDual(-r, -d, -h);
    }


    /**
     * @brief Element-wise subtraction of two TensorHyperDual objects.
     */
    TensorHyperDual operator-(const TensorHyperDual& other) const {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator- – dtype or device mismatch");

        return TensorHyperDual(r - other.r,
                            d - other.d,
                            h - other.h);
    }


    /**
     * @brief Element-wise product of two TensorHyperDual objects.
     *
     * r_new = r₁ r₂
     * d_new = r₁ d₂ + d₁ r₂
     * h_new = d₁⊗d₂ + d₂⊗d₁ + r₁ h₂ + r₂ h₁
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual operator*(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator* – dtype or device mismatch");

        // ─── Real part ──────────────────────────────────────────────────────────
        auto rn = r * other.r;                                    // [N]

        // ─── Dual part: d_new = r₁ d₂ + d₁ r₂ ──────────────────────────────────
        auto dn = r.unsqueeze(1) * other.d                       // broadcast [N,1]×[N,D]
                + other.r.unsqueeze(1) * d;                      // [N,D]

        // ─── Hyper-dual part ───────────────────────────────────────────────────
        //   d⊗d terms
        auto outer12 = d.unsqueeze(2) * other.d.unsqueeze(1);    // [N,D,D]
        auto outer21 = other.d.unsqueeze(2) * d.unsqueeze(1);    // [N,D,D]

        //   r h terms
        auto rh2 = r.unsqueeze(1).unsqueeze(2) * other.h;        // [N,D,D]
        auto r2h1 = other.r.unsqueeze(1).unsqueeze(2) * h;       // [N,D,D]

        auto hn = outer12 + outer21 + rh2 + r2h1;                // [N,D,D]

        return TensorHyperDual(rn, dn, hn);
    }


    /**
     * @brief Element-wise division of two TensorHyperDual objects: *this / other.
     *
     * r_new = r₁ / r₂
     * d_new = (d₁ r₂ − r₁ d₂) / r₂²
     * h_new =  h₁ / r₂
     *        − r₁ h₂ / r₂²
     *        − 2 d₁⊗d₂ / r₂²
     *        + 2 r₁ (d₂⊗d₂) / r₂³
     *
     * Shapes
     *   r : [N],   d : [N,D],   h : [N,D,D]
     */
    TensorHyperDual operator/(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator/ – dtype or device mismatch");

        /*────────── scalar helpers ──────────*/
        auto inv   = other.r.reciprocal();          // 1 / r₂            [N]
        auto inv2  = inv * inv;                     // 1 / r₂²           [N]
        auto inv3  = inv2 * inv;                    // 1 / r₂³           [N]

        /*────────── real part ───────────────*/
        auto rn = r * inv;                          // r₁ / r₂           [N]

        /*────────── dual part ───────────────*/
        auto numer_d = d * other.r.unsqueeze(1)     // d₁ r₂
                    - r.unsqueeze(1) * other.d;    // r₁ d₂
        auto dn = numer_d * inv2.unsqueeze(1);      // / r₂²             [N,D]

        /*────────── hyper-dual part ─────────*/
        //  h₁ / r₂
        auto term_h1 = h * inv.unsqueeze(1).unsqueeze(2);         // [N,D,D]

        // − r₁ h₂ / r₂²
        auto term_h2 = - r.unsqueeze(1).unsqueeze(2)
                    * other.h * inv2.unsqueeze(1).unsqueeze(2); // [N,D,D]

        // − 2 d₁⊗d₂ / r₂²
        auto outer12 = d.unsqueeze(2) * other.d.unsqueeze(1);      // d₁⊗d₂
        auto term_h3 = -2.0 * outer12 * inv2.unsqueeze(1).unsqueeze(2);

        // + 2 r₁ (d₂⊗d₂) / r₂³
        auto d2_outer = other.d.unsqueeze(2) * other.d.unsqueeze(1); // d₂⊗d₂
        auto term_h4  =  2.0 * r.unsqueeze(1).unsqueeze(2)
                            * d2_outer * inv3.unsqueeze(1).unsqueeze(2);

        auto hn = term_h1 + term_h2 + term_h3 + term_h4;          // [N,D,D]

        return TensorHyperDual(rn, dn, hn);
    }



    /**
     * @brief Element-wise division by a plain torch::Tensor.
     *
     * * other must be broadcast-compatible with `r` (shape `[N]` or scalar).
     *
     * r_new = r  / other
     * d_new = d  / other
     * h_new = h  / other
     */
    TensorHyperDual operator/(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        // Promote `other` to shape [N] (or keep scalar) for broadcasting
        auto denom = other.squeeze();                         // allow scalar
        if (denom.dim() == 0)        denom = denom.expand_as(r);   // scalar → [N]
        TORCH_CHECK(denom.sizes() == r.sizes(),
                    "other must be scalar or of shape [N]");

        // Add tiny ε to avoid singularity when requested
        constexpr double eps = 1e-14;
        auto safe_denom = denom + (denom.abs() < eps).to(denom.dtype()) * eps;

        auto inv   = safe_denom.reciprocal();                 // 1 / denom   [N]
        auto inv_d = inv.unsqueeze(1);                        // [N,1]
        auto inv_h = inv.unsqueeze(1).unsqueeze(2);           // [N,1,1]

        return TensorHyperDual(r * inv,                      // r / denom   [N]
                            d * inv_d,                    // d / denom   [N,D]
                            h * inv_h);                   // h / denom   [N,D,D]
    }

    /**
     * @brief Divide a TensorHyperDual by a scalar.
     *
     * @param scalar  Real (or complex) scalar.  Must be non-zero.
     * @return TensorHyperDual with every block divided by @p scalar.
     */
    TensorHyperDual operator/(double scalar) const
    {
        TORCH_CHECK(scalar != 0.0, "TensorHyperDual::operator/: division by zero");

        // torch handles real→complex promotion automatically if r is complex
        return TensorHyperDual(r / scalar,
                            d / scalar,
                            h / scalar);
    }


    /**
     * @brief Element-wise ≤ comparison (real part only).
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *   true  ⇔  r[i] ≤ other.r[i]
     */
    torch::Tensor operator<=(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator<= – dtype or device mismatch");

        return r <= other.r;      // bool tensor [N]
    }


    /**
     * @brief Element-wise equality test on the real part.
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *   true ⇔ r[i] == other.r[i].
     *
     * If you later want full structural equality (r, d, h all equal),
     * just replace the body with
     *   return (r == other.r) & (d == other.d) & (h == other.h);
     */
    torch::Tensor operator==(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator== – dtype or device mismatch");

        return r == other.r;   // bool tensor [N]
    }

    /**
     * @brief Compare the real part of a TensorHyperDual with a torch::Tensor (or scalar).
     *
     * Returns a boolean tensor whose shape is the broadcast of `r` and `other`.
     */
    torch::Tensor operator<=(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        /*  PyTorch handles broadcasting automatically, so this works for
            * scalars,
            * tensors of shape [N] (matching r), or
            * any other broadcast-compatible shape.                            */
        return r <= other;   // bool tensor
    }


    /**
     * @brief Compare the real part of a TensorHyperDual with a scalar.
     *
     * Accepts any arithmetic type (int, float, double, etc.).
     * Returns a boolean tensor of shape [N].
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator<=(Scalar scalar) const
    {
        // Broadcasting happens automatically (scalar → [N])
        return r <= scalar;     // bool tensor [N]
    }

    /**
     * @brief Element-wise ‘greater than’ comparison of two TensorHyperDual objects
     *        (real part only).
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *     true ⇔ r[i] > other.r[i].
     */
    torch::Tensor operator>(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator> – dtype or device mismatch");

        return r > other.r;           // bool tensor [N]
    }


    /**
     * @brief Compare the real part with a scalar:  *this > scalar
     *
     * Accepts any arithmetic type (int, float, double, …).
     * Returns a bool tensor of shape [N].
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator>(Scalar scalar) const
    {
        return r > scalar;          // broadcasts scalar → [N]
    }

    /**
     * @brief Compare the real part with a scalar:  *this > scalar
     *
     * Works for any arithmetic type (int, float, double, …).
     * Relies on PyTorch’s automatic scalar broadcasting.
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator>(Scalar scalar) const
    {
        return r > scalar;     // bool tensor [N]  (scalar → broadcast)
    }

    /**
     * @brief Element-wise ‘less than’ comparison of two TensorHyperDual objects
     *        (real part only).
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *     true ⇔ r[i] < other.r[i].
     */
    torch::Tensor operator<(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator< – dtype or device mismatch");

        return r < other.r;      // bool tensor [N]
    }

    /**
     * @brief Compare the real part of a TensorHyperDual with a torch::Tensor (or scalar):
     *        *this < other
     *
     * Accepts a scalar, a tensor of shape `[N]`, or any broadcast-compatible shape.
     * Returns a boolean tensor with the broadcasted shape.
     */
    torch::Tensor operator<(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        // Broadcasting handles scalar or vector automatically.
        return r < other;     // bool tensor
    }

    /**
     * @brief Compare the real part with a scalar: *this < scalar
     *
     * Works for any arithmetic type (`int`, `float`, `double`, …).
     * The scalar is broadcast by PyTorch to match the shape `[N]`.
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator<(Scalar scalar) const
    {
        return r < scalar;        // bool tensor [N]
    }



    /**
     * @brief Element-wise ‘greater than or equal’ comparison of two
     *        TensorHyperDual objects (real part only).
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *     true ⇔ r[i] ≥ other.r[i].
     */
    torch::Tensor operator>=(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator>= – dtype or device mismatch");

        return r >= other.r;     // bool tensor [N]
    }

    /**
     * @brief Compare the real part of a TensorHyperDual with a torch::Tensor (or scalar):
     *        *this >= other
     *
     * Accepts a scalar, a tensor of shape `[N]`, or any broadcast-compatible shape.
     * Returns a boolean tensor with the broadcasted shape.
     */
    torch::Tensor operator>=(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        // Broadcasting covers scalar, vector, etc.
        return r >= other;      // bool tensor
    }

    /**
     * @brief Compare the real part with a scalar:  *this >= scalar
     *
     * Works for any arithmetic type (int, float, double, …).
     * Returns a boolean tensor of shape [N].
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator>=(Scalar scalar) const
    {
        return r >= scalar;          // bool tensor [N]
    }



    /**
     * @brief Compare the real part of a TensorHyperDual with a torch::Tensor (or scalar):
     *        *this == other
     *
     * Accepts a scalar, a tensor of shape `[N]`, or any broadcast-compatible shape.
     * Returns a boolean tensor with the broadcasted shape.
     */
    torch::Tensor operator==(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        // PyTorch handles scalar / vector broadcasting automatically.
        return r.eq(other);     // bool tensor
    }

    
    /**
     * @brief Compare the real part with a scalar:  *this == scalar
     *
     * Works for any arithmetic type (`int`, `float`, `double`, …).
     * Returns a boolean tensor of shape [N].
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator==(Scalar scalar) const
    {
        // Broadcasting turns the scalar into shape [N] automatically.
        return r.eq(scalar);          // bool tensor [N]
    }

    /**
     * @brief Element-wise inequality comparison (real part only) between two
     *        TensorHyperDual objects.
     *
     * Returns a boolean tensor of shape [N] whose i-th entry is
     *     true ⇔ r[i] != other.r[i].
     */
    torch::Tensor operator!=(const TensorHyperDual& other) const
    {
        TORCH_CHECK(r.dtype()  == other.r.dtype()  &&
                    r.device() == other.r.device(),
                    "TensorHyperDual::operator!= – dtype or device mismatch");

        return r.ne(other.r);    // bool tensor [N]
    }

    /**
     * @brief Compare the real part of a TensorHyperDual with a torch::Tensor (or scalar):
     *        *this != other
     *
     * Accepts a scalar, a tensor of shape `[N]`, or any broadcast-compatible shape.
     * Returns a boolean tensor with the broadcasted shape.
     */
    torch::Tensor operator!=(const torch::Tensor& other) const
    {
        TORCH_CHECK(other.dtype()  == r.dtype(),   "dtype mismatch");
        TORCH_CHECK(other.device() == r.device(),  "device mismatch");

        // PyTorch handles scalar / vector broadcasting automatically.
        return r.ne(other);      // bool tensor
    }
    

    /**
     * @brief Compare the real part with a scalar:  *this != scalar
     *
     * Works for any arithmetic type (int, float, double, …).
     * Returns a boolean tensor of shape [N].
     */
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    torch::Tensor operator!=(Scalar scalar) const
    {
        return r.ne(scalar);        // bool tensor [N]  (scalar broadcasted)
    }

    /**
     * @brief Element-wise reciprocal:  y = 1 / x
     *
     * r_new = 1 / r
     * d_new = − d / r²
     * h_new = 2 (d⊗d) / r³  −  h / r²
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual reciprocal() const
    {
        /*────────── helpers ──────────*/
        auto inv   = r.reciprocal();           // 1/r          [N]
        auto inv2  = inv * inv;                // 1/r²         [N]
        auto inv3  = inv2 * inv;               // 1/r³         [N]

        /*────────── real part ─────────*/
        auto rn = inv;                         // [N]

        /*────────── dual part ─────────*/
        auto dn = -d * inv2.unsqueeze(1);      // −d / r²      [N,D]

        /*────────── hyper-dual part ───*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);           // d⊗d   [N,D,D]
        auto hn =  2.0 * outer * inv3.unsqueeze(1).unsqueeze(2) // 2 d⊗d / r³
                -       h * inv2.unsqueeze(1).unsqueeze(2);    // − h / r²

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise cosine of a TensorHyperDual.
     *
     * r_new =  cos r
     * d_new = −sin r · d
     * h_new = −cos r · (d ⊗ d)  −  sin r · h
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual cos() const
    {
        /*────────── trigonometric helpers ──────────*/
        auto c = torch::cos(r);   // cos r   [N]
        auto s = torch::sin(r);   // sin r   [N]

        /*────────── real part ─────────*/
        auto rn = c;                                  // [N]

        /*────────── dual part ─────────*/
        auto dn = -s.unsqueeze(1) * d;                // −sin r · d   [N,D]

        /*────────── hyper-dual part ───*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1); // d ⊗ d       [N,D,D]

        auto hn = -c.unsqueeze(1).unsqueeze(2) * outer   // −cos r · d⊗d
                -s.unsqueeze(1).unsqueeze(2) * h;      // −sin r · h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise sine of a TensorHyperDual.
     *
     * r_new =  sin r
     * d_new =  cos r · d
     * h_new = −sin r · (d ⊗ d)  +  cos r · h
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual sin() const
    {
        /*────────── trig helpers ──────────*/
        auto s = torch::sin(r);   // sin r   [N]
        auto c = torch::cos(r);   // cos r   [N]

        /*────────── real part ─────────────*/
        auto rn = s;                                  // [N]

        /*────────── dual part ─────────────*/
        auto dn = c.unsqueeze(1) * d;                 // cos r · d       [N,D]

        /*────────── hyper-dual part ───────*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1); // d ⊗ d           [N,D,D]

        auto hn = -s.unsqueeze(1).unsqueeze(2) * outer   // −sin r · d⊗d
                + c.unsqueeze(1).unsqueeze(2) * h;     // +cos r · h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise tangent of a TensorHyperDual.
     *
     * r_new =  tan r
     * d_new =  sec² r · d
     * h_new =  2 sec² r·tan r · (d ⊗ d)  +  sec² r · h
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual tan() const
    {
        /*────────── helper scalars ──────────*/
        auto t   = torch::tan(r);                  // tan r     [N]
        auto sec2= torch::cos(r).pow(-2);          // sec² r    [N]  (1/ cos²)

        /*────────── real part ───────────────*/
        auto rn = t;                               // [N]

        /*────────── dual part ───────────────*/
        auto dn = sec2.unsqueeze(1) * d;           // sec² r · d        [N,D]

        /*────────── hyper-dual part ─────────*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);  // d ⊗ d         [N,D,D]

        auto hn = 2.0 * (sec2 * t).unsqueeze(1).unsqueeze(2) * outer   // 2 sec² r tan r · d⊗d
                +        sec2.unsqueeze(1).unsqueeze(2) * h;           // + sec² r · h

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * @brief Element-wise arcsine of a TensorHyperDual.
     *
     * r_new =  asin r
     * d_new =  d / √(1 − r²)
     * h_new =  r · (d⊗d) / (1 − r²)^{3/2}  +  h / √(1 − r²)
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual asin() const
    {
        /*── Promote to complex if any |r|>1 and tensor is real ────────────────*/
        torch::Tensor rc = r, dc = d, hc = h;
        if (!r.is_complex() && (r.abs() > 1).any().item<bool>()) {
            auto z = torch::zeros_like(r);
            rc = torch::complex(r, z);
            dc = torch::complex(d, z.unsqueeze(1));
            hc = torch::complex(h, z.unsqueeze(1).unsqueeze(2));
        }

        /*── Helper scalars ────────────────────────────────────────────────────*/
        auto one_minus_r2 = 1 - rc * rc;              // 1 − r²          [N]
        auto inv_sqrt     = one_minus_r2.rsqrt();      // 1 / √(1−r²)     [N]
        auto inv_sqrt3    = inv_sqrt / one_minus_r2;   // 1/(1−r²)^{3/2}  [N]
        auto ypp          = rc * inv_sqrt3;            // r / (1−r²)^{3/2}[N]

        /*── Real part ─────────────────────────────────────────────────────────*/
        auto rn = torch::asin(rc);                     // asin r          [N]

        /*── Dual part:  d / √(1−r²) ───────────────────────────────────────────*/
        auto dn = dc * inv_sqrt.unsqueeze(1);          // [N,D]

        /*── Hyper-dual part ───────────────────────────────────────────────────*/
        auto outer = dc.unsqueeze(2) * dc.unsqueeze(1);        // d⊗d   [N,D,D]

        auto hn = ypp.unsqueeze(1).unsqueeze(2) * outer         // r·d⊗d / (1−r²)^{3/2}
                + inv_sqrt.unsqueeze(1).unsqueeze(2) * hc;      // + h / √(1−r²)

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * @brief Element-wise arccosine of a TensorHyperDual.
     *
     * r_new =  acos r
     * d_new = − d / √(1 − r²)
     * h_new = − r · (d⊗d) / (1 − r²)^{3/2}  −  h / √(1 − r²)
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual acos() const
    {
        /*── Promote to complex if |r|>1 and tensor is real ────────────────────*/
        torch::Tensor rc = r, dc = d, hc = h;
        if (!r.is_complex() && (r.abs() > 1).any().item<bool>()) {
            auto z = torch::zeros_like(r);
            rc = torch::complex(r, z);
            dc = torch::complex(d, z.unsqueeze(1));
            hc = torch::complex(h, z.unsqueeze(1).unsqueeze(2));
        }

        /*── Helper scalars ────────────────────────────────────────────────────*/
        auto one_minus_r2 = 1 - rc * rc;              // 1 − r²            [N]
        auto inv_sqrt     = one_minus_r2.rsqrt();      // 1 / √(1−r²)       [N]
        auto inv_sqrt3    = inv_sqrt / one_minus_r2;   // 1/(1−r²)^{3/2}    [N]
        auto ypp          = rc * inv_sqrt3;            // r/(1−r²)^{3/2}    [N]

        /*── Real part ─────────────────────────────────────────────────────────*/
        auto rn = torch::acos(rc);                     // acos r            [N]

        /*── Dual part:  −d / √(1−r²) ──────────────────────────────────────────*/
        auto dn = -dc * inv_sqrt.unsqueeze(1);         // [N,D]

        /*── Hyper-dual part ───────────────────────────────────────────────────*/
        auto outer = dc.unsqueeze(2) * dc.unsqueeze(1);      // d⊗d   [N,D,D]

        auto hn = -ypp.unsqueeze(1).unsqueeze(2) * outer      // −r·d⊗d /(1−r²)^{3/2}
                -inv_sqrt.unsqueeze(1).unsqueeze(2) * hc;   // −h / √(1−r²)

        return TensorHyperDual(rn, dn, hn);
    }
    
    /**
     * @brief Element-wise arctangent of a TensorHyperDual.
     *
     * r_new =  atan r
     * d_new =  d / (1 + r²)
     * h_new = − 2 r · (d ⊗ d) / (1 + r²)²  +  h / (1 + r²)
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual atan() const
    {
        /*── Helper scalars ────────────────────────────────────────────────────*/
        auto one_plus_r2 = 1 + r * r;             // 1 + r²           [N]
        auto inv1        = one_plus_r2.reciprocal();      // 1 / (1+r²)   [N]
        auto inv2        = inv1 * inv1;                   // 1 / (1+r²)²  [N]

        /*── Real part ─────────────────────────────────────────────────────────*/
        auto rn = torch::atan(r);                 // atan r           [N]

        /*── Dual part  d / (1+r²) ─────────────────────────────────────────────*/
        auto dn = d * inv1.unsqueeze(1);          // [N,D]

        /*── Hyper-dual part ───────────────────────────────────────────────────*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);          // d⊗d   [N,D,D]

        auto hn = -2.0 * r.unsqueeze(1).unsqueeze(2) * outer * inv2.unsqueeze(1).unsqueeze(2)
                +       h * inv1.unsqueeze(1).unsqueeze(2);  // [N,D,D]

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise hyperbolic sine of a TensorHyperDual.
     *
     * r_new =  sinh r
     * d_new =  cosh r · d
     * h_new =  sinh r · (d ⊗ d)  +  cosh r · h
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual sinh() const
    {
        /*────────── helpers ──────────*/
        auto sh = torch::sinh(r);     // sinh r   [N]
        auto ch = torch::cosh(r);     // cosh r   [N]

        /*────────── real part ────────*/
        auto rn = sh;                                   // [N]

        /*────────── dual part ────────*/
        auto dn = ch.unsqueeze(1) * d;                  // cosh r · d     [N,D]

        /*────────── hyper-dual part ──*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);   // d ⊗ d          [N,D,D]

        auto hn = sh.unsqueeze(1).unsqueeze(2) * outer   // sinh r · d⊗d
                + ch.unsqueeze(1).unsqueeze(2) * h;      // + cosh r · h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise hyperbolic cosine of a TensorHyperDual.
     *
     * r_new =  cosh r
     * d_new =  sinh r · d
     * h_new =  cosh r · (d ⊗ d)  +  sinh r · h
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual cosh() const
    {
        /*────────── helpers ──────────*/
        auto ch = torch::cosh(r);     // cosh r   [N]
        auto sh = torch::sinh(r);     // sinh r   [N]

        /*────────── real part ────────*/
        auto rn = ch;                                   // [N]

        /*────────── dual part ────────*/
        auto dn = sh.unsqueeze(1) * d;                  // sinh r · d     [N,D]

        /*────────── hyper-dual part ──*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);   // d ⊗ d          [N,D,D]

        auto hn = ch.unsqueeze(1).unsqueeze(2) * outer   // cosh r · d⊗d
                + sh.unsqueeze(1).unsqueeze(2) * h;      // + sinh r · h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise hyperbolic tangent of a TensorHyperDual.
     *
     * r_new =  tanh r
     * d_new =  sech² r · d
     * h_new = −2 sech² r·tanh r · (d ⊗ d)  +  sech² r · h
     *
     * where  sech² r = 1 / cosh² r .
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual tanh() const
    {
        /*────────── helpers ──────────*/
        auto t     = torch::tanh(r);             // tanh r              [N]
        auto sech2 = torch::cosh(r).pow(-2);     // 1 / cosh² r         [N]

        /*────────── real part ────────*/
        auto rn = t;                             // [N]

        /*────────── dual part ────────*/
        auto dn = sech2.unsqueeze(1) * d;        // sech² r · d         [N,D]

        /*────────── hyper-dual part ──*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);          // d ⊗ d  [N,D,D]

        auto hn = -2.0 * (sech2 * t).unsqueeze(1).unsqueeze(2) * outer   // −2 sech² r tanh r · d⊗d
                +       sech2.unsqueeze(1).unsqueeze(2) * h;           // + sech² r · h

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise exponential of a TensorHyperDual.
     *
     * r_new =  eʳ
     * d_new =  eʳ · d
     * h_new =  eʳ · (d ⊗ d + h)
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual exp() const
    {
        /*────────── real part ───────────────*/
        auto rn = torch::exp(r);                         // eʳ           [N]

        /*────────── dual part ───────────────*/
        auto dn = rn.unsqueeze(1) * d;                   // eʳ · d       [N,D]

        /*────────── hyper-dual part ─────────*/
        auto outer = d.unsqueeze(2) * d.unsqueeze(1);    // d ⊗ d       [N,D,D]
        auto hn = rn.unsqueeze(1).unsqueeze(2) * (outer + h);   // eʳ·(d⊗d + h)

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Element-wise natural logarithm of a TensorHyperDual.
     *
     * r_new =  ln r
     * d_new =  d / r
     * h_new = − (d ⊗ d) / r²  +  h / r
     *
     * Shapes
     *   r : [N]          d : [N,D]          h : [N,D,D]
     */
    TensorHyperDual log() const
    {
        /*── Promote to complex if any r ≤ 0 and tensor is real ────────────────*/
        torch::Tensor rc = r, dc = d, hc = h;
        if (!r.is_complex() && (r <= 0).any().item<bool>()) {
            auto z = torch::zeros_like(r);
            rc = torch::complex(r, z);
            dc = torch::complex(d, z.unsqueeze(1));
            hc = torch::complex(h, z.unsqueeze(1).unsqueeze(2));
        }

        /*── Helper scalars ────────────────────────────────────────────────────*/
        auto inv  = rc.reciprocal();          // 1 / r           [N]
        auto inv2 = inv * inv;                // 1 / r²          [N]

        /*── Real part ─────────────────────────────────────────────────────────*/
        auto rn = torch::log(rc);             // ln r            [N]

        /*── Dual part  d / r ─────────────────────────────────────────────────*/
        auto dn = dc * inv.unsqueeze(1);      // [N,D]

        /*── Hyper-dual part ──────────────────────────────────────────────────*/
        auto outer = dc.unsqueeze(2) * dc.unsqueeze(1);          // d⊗d   [N,D,D]

        auto hn = - outer * inv2.unsqueeze(1).unsqueeze(2)        // − d⊗d / r²
                + hc   * inv .unsqueeze(1).unsqueeze(2);        // + h / r

        return TensorHyperDual(rn, dn, hn);
    }


    /**
     * @brief Element-wise absolute value of a TensorHyperDual.
     *
     * For real r:
     *   r_new = |r|
     *   d_new = sign(r) · d
     *   h_new = sign(r) · h
     *
     * For complex r: we define “sign” as r / |r| so the C-R rules remain valid.
     */
    TensorHyperDual abs() const
    {
        // ── magnitude ─────────────────────────────────────────────────────────
        auto mag = torch::abs(r);                       // |r|          [N]

        // ── sign factor ───────────────────────────────────────────────────────
        torch::Tensor s;
        if (r.is_complex())
            s = r / mag;                                // r / |r|      [N]
        else
            s = torch::sgn(r);                          // sign(r)      [N]

        // ── assemble components ───────────────────────────────────────────────
        auto rn = mag;                                  // [N]
        auto dn = s.unsqueeze(1)               * d;     // sign · d     [N,D]
        auto hn = s.unsqueeze(1).unsqueeze(2) * h;      // sign · h     [N,D,D]

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Return a complex‐typed copy of this TensorHyperDual.
     *
     * If the object is already complex, the existing tensors are reused.
     * Otherwise real tensors are promoted via torch::complex(real, 0).
     */
    TensorHyperDual toComplex() const
    {
        // Real block
        torch::Tensor rc = r.is_complex()
                        ? r
                        : torch::complex(r, torch::zeros_like(r));

        // Dual block
        torch::Tensor dc = d.is_complex()
                        ? d
                        : torch::complex(d, torch::zeros_like(d));

        // Hyper-dual block
        torch::Tensor hc = h.is_complex()
                        ? h
                        : torch::complex(h, torch::zeros_like(h));

        return TensorHyperDual(rc, dc, hc);
    }

    /**
     * @brief Extract the real part of every block.
     *
     * If the TensorHyperDual is already real (`!r.is_complex()`), the
     * current tensors are reused; otherwise we materialize torch::real().
     */
    TensorHyperDual real() const
    {
        if (!r.is_complex()) {
            // Already real – just return a shallow copy.
            return *this;
        }

        // Promote complex → real
        return TensorHyperDual(torch::real(r),
                            torch::real(d),
                            torch::real(h));
    }

    /**
     * @brief Extract the imaginary part of each block.
     *
     * For real-typed tensors the imaginary part is zero, so we return a
     * matching `zeros_like`.  For complex tensors we return `torch::imag`.
     */
    TensorHyperDual imag() const
    {
        auto rn = r.is_complex() ? torch::imag(r)
                                : torch::zeros_like(r);

        auto dn = d.is_complex() ? torch::imag(d)
                                : torch::zeros_like(d);

        auto hn = h.is_complex() ? torch::imag(h)
                                : torch::zeros_like(h);

        return TensorHyperDual(rn, dn, hn);
    }

    /**
     * @brief Return a TensorHyperDual whose r, d, h blocks are all zeros,
     *        matching this object’s shape, dtype, and device.
     */
    TensorHyperDual zeros_like() const
    {
        return TensorHyperDual(torch::zeros_like(r),
                            torch::zeros_like(d),
                            torch::zeros_like(h));
    }


    /**
     * @brief Element-wise sign of a TensorHyperDual.
     *
     * For real inputs, this is `sgn(r)` ∈ {-1, 0, 1}.  
     * For complex inputs we follow PyTorch’s convention  
     *  sgn(z) = z / |z|   (returns 0 when z == 0).
    *
    * Because sign is constant almost everywhere, its first- and second-order
    * derivatives are zero, so the dual and hyper-dual blocks become zeros.
    */
    TensorHyperDual sign() const
    {
        // Real block – sign definition depends on dtype
        auto sign_r = torch::sgn(r);                 // works for real & complex

        // Derivative blocks – identically zero
        auto sign_d = torch::zeros_like(d);          // [N,D]
        auto sign_h = torch::zeros_like(h);          // [N,D,D]

        return TensorHyperDual(sign_r, sign_d, sign_h);
    }
    
    /**
     * @brief Return the element with minimal real part.
     *
     * r : [N]          d : [N,D]          h : [N,D,D]
     *
     * After the call the result has
     *   r : [1]        d : [1,D]          h : [1,D,D]
     */
    TensorHyperDual min() const
    {
        // ── locate arg-min on axis 0 ───────────────────────────────────────────
        auto idx = torch::argmin(r, /*dim=*/0, /*keepdim=*/false);   // scalar

        // ── gather matching blocks ────────────────────────────────────────────
        auto r_min = r.index({idx}).unsqueeze(0);                    // [1]
        auto d_min = d.index({idx}).unsqueeze(0);                    // [1,D]
        auto h_min = h.index({idx}).unsqueeze(0);                    // [1,D,D]

        return TensorHyperDual(r_min, d_min, h_min);
    }
    
    /**
     * @brief Return the element with maximal real part.
     *
     * Current layout
     *   r : [N]          d : [N,D]          h : [N,D,D]
     *
     * Result layout
     *   r : [1]          d : [1,D]          h : [1,D,D]
     */
    TensorHyperDual max() const
    {
        // ── index of the maximum along the state axis ────────────────────────
        auto idx = torch::argmax(r, /*dim=*/0, /*keepdim=*/false);   // scalar

        // ── gather matching blocks ───────────────────────────────────────────
        auto r_max = r.index({idx}).unsqueeze(0);                    // [1]
        auto d_max = d.index({idx}).unsqueeze(0);                    // [1,D]
        auto h_max = h.index({idx}).unsqueeze(0);                    // [1,D,D]

        return TensorHyperDual(r_max, d_max, h_max);
    }

    /**
     * @brief Element-wise selection for TensorHyperDual:
     *        result = cond ? x : y
     *
     * @param cond  Boolean mask.  Accepts
     *                • scalar (0-D),
     *                • shape [N]       – matches r,
     *                • shape [N,D]     – matches d,
     *                • shape [N,D,D]   – matches h.
     *              Broadcasting is applied automatically.
     * @param x, y  TensorHyperDual objects with identical shapes / dtype / device.
     */
    static TensorHyperDual where(const torch::Tensor& cond,
        const TensorHyperDual& x,
        const TensorHyperDual& y)
    {
        // ── sanity checks ─────────────────────────────────────────────────────
        TORCH_CHECK(x.r.sizes() == y.r.sizes() &&
        x.d.sizes() == y.d.sizes() &&
        x.h.sizes() == y.h.sizes(),
        "TensorHyperDual::where – x and y shape mismatch");

        TORCH_CHECK(cond.dtype() == torch::kBool,
        "TensorHyperDual::where – cond must be boolean");

        // ── broadcast mask to each block’s shape ──────────────────────────────
        auto mask_r = cond;                                   // [N] or broadcastable
        auto mask_d = cond.dim() == 0 ? cond                  // scalar
        : cond.squeeze().unsqueeze(1);          // [N,1]  → broadcast to [N,D]
        auto mask_h = cond.dim() == 0 ? cond
        : cond.squeeze().unsqueeze(1).unsqueeze(2); // [N,1,1]

        // ── select ────────────────────────────────────────────────────────────
        auto r_sel = torch::where(mask_r, x.r, y.r);          // [N]
        auto d_sel = torch::where(mask_d, x.d, y.d);          // [N,D]
        auto h_sel = torch::where(mask_h, x.h, y.h);          // [N,D,D]

        return TensorHyperDual(r_sel, d_sel, h_sel);
    }

    /**
     * @brief Limited two-operand einsum for TensorHyperDual.
     *
     * Only supports the signature "subscripts,subscripts->output".
     * Shapes must be broadcast-compatible in the same way PyTorch expects.
     */
    static TensorHyperDual einsum(const std::string& subscripts,
        const TensorHyperDual& a,
        const TensorHyperDual& b)
    {
        // ── real part ─────────────────────────────────────────────────────────
        auto r_out = torch::einsum(subscripts, {a.r, b.r});

        // ── build modified subscripts once to avoid string allocations ───────
        // e.g.  "ij,jk->ik"  becomes  "ij,jkz->ikz" etc.
        const auto comma = subscripts.find(',');
        const auto arrow = subscripts.find("->");

        std::string lhsA  = subscripts.substr(0, comma);               // "ij"
        std::string lhsB  = subscripts.substr(comma + 1,
                        arrow - comma - 1);      // "jk"
        std::string rhs   = subscripts.substr(arrow + 2);              // "ik"

        // helper to append extra index letters
        auto append = [](std::string base, const char* extra) {
            base.append(extra);
            return base;
       };

        // d-part:  A⊗d_B  and  d_A⊗B        (add new trailing dim 'z')
        auto d_aB = torch::einsum( append(lhsA,         "" ) + "," +
            append(lhsB,    "z") + "->" +
            append(rhs, "z"),
            {a.r, b.d});

        auto d_Ab = torch::einsum( append(lhsA,    "z") + "," +
            append(lhsB,         "" ) + "->" +
            append(rhs, "z"),
            {a.d, b.r});

        auto d_out = d_aB + d_Ab;                                    // [*,  D]

        // h-part:  A⊗h_B ,  2 d_A⊗d_B ,  h_A⊗B      (new dims 'z','w')
        auto h_aB = torch::einsum( append(lhsA,        "" ) + "," +
            append(lhsB, "zw") + "->" +
            append(rhs, "zw"),
            {a.r, b.h});

        auto h_dAdB = torch::einsum( append(lhsA,   "z") + "," +
            append(lhsB,   "w") + "->" +
            append(rhs,"zw"),
            {a.d, b.d});

        auto h_Ab = torch::einsum( append(lhsA,"zw") + "," +
            append(lhsB,   "" ) + "->" +
            append(rhs,"zw"),
            {a.h, b.r});

        auto h_out = h_aB + 2.0 * h_dAdB + h_Ab;                     // [*, D, D]

        return TensorHyperDual(r_out, d_out, h_out);
    }



    /**
     * @brief Two-operand einsum where the *first* argument is an ordinary tensor
     *        (no derivatives) and the *second* is a TensorHyperDual.
     *
     * Given subscripts `"A,B->C"`:
     *     r_out = einsum(A, r_B)
     *     d_out = einsum(A, d_B)
     *     h_out = einsum(A, h_B)
     *
     * The operation is effectively linear in the second operand, so the
     * first tensor contributes only to the scalar multiplier.
     */
    static TensorHyperDual einsum(const std::string& subscripts,
        const torch::Tensor& a,          // no derivatives
        const TensorHyperDual& b)        // has r, d, h
    {
        // real part
        auto r_out = torch::einsum(subscripts, {a, b.r});

        // create new subscripts once for d and h
        auto comma = subscripts.find(',');
        auto arrow = subscripts.find("->");

        std::string lhsA = subscripts.substr(0, comma);                 // "A"
        std::string lhsB = subscripts.substr(comma + 1,
                    arrow - comma - 1);        // "B"
        std::string rhs  = subscripts.substr(arrow + 2);                // "C"

        // d_out  : add a trailing index 'z'
        auto subs_d = lhsA + "," + lhsB + "z->" + rhs + "z";
        auto d_out  = torch::einsum(subs_d, {a, b.d});                  // A ⊗ d_B

        // h_out  : add trailing indices 'z','w'
        auto subs_h = lhsA + "," + lhsB + "zw->" + rhs + "zw";
        auto h_out  = torch::einsum(subs_h, {a, b.h});                  // A ⊗ h_B

        return TensorHyperDual(r_out, d_out, h_out);
    }


    /**
     * @brief Two-operand einsum where the first input has derivatives and the
     *        second is a plain tensor (no derivatives).
     *
     * Subscripts: "A,B->C"
     *   r_out = einsum(r_A , B)
     *   d_out = einsum(d_A , B)
     *   h_out = einsum(h_A , B)
     */
    static TensorHyperDual einsum(const std::string& subscripts,
        const TensorHyperDual& a,   // has r,d,h
        const torch::Tensor& b)      // plain tensor
    {
        // real part
        auto r_out = torch::einsum(subscripts, {a.r, b});

        // split subscripts once
        auto comma = subscripts.find(',');
        auto arrow = subscripts.find("->");

        std::string lhsA = subscripts.substr(0, comma);                 // "A"
        std::string lhsB = subscripts.substr(comma + 1,
                    arrow - comma - 1);        // "B"
        std::string rhs  = subscripts.substr(arrow + 2);                // "C"

        // d_out : add trailing index 'z'
        auto subs_d = lhsA + "z," + lhsB + "->" + rhs + "z";
        auto d_out  = torch::einsum(subs_d, {a.d, b});                  // d_A ⊗ B

        // h_out : add trailing indices 'z','w'
        auto subs_h = lhsA + "zw," + lhsB + "->" + rhs + "zw";
        auto h_out  = torch::einsum(subs_h, {a.h, b});                  // h_A ⊗ B

        return TensorHyperDual(r_out, d_out, h_out);
    }

    static TensorHyperDual einsum(const std::string &subs,
                                  const std::vector<TensorHyperDual> &xs)
    {
        TORCH_CHECK(xs.size() >= 2,
                    "TensorHyperDual::einsum – need at least two operands");

        // Start with the first argument’s real part
        auto result_r = torch::einsum(subs,
                                      {xs[0].r, xs[1].r}); // plain real einsum

        auto result_d = torch::zeros_like(xs[0].d);
        auto result_h = torch::zeros_like(xs[0].h);

        /*──────────────────────── accumulate first-order terms ────────────────*/
        for (size_t i = 0; i < xs.size(); ++i)
        {
            // Build operand list:       r_0 , ... , r_{i-1} , d_i , r_{i+1} , ...
            std::vector<torch::Tensor> args;
            args.reserve(xs.size());

            std::string lhs_subs;
            size_t pos = 0, next;
            size_t lhs_idx = 0;

            /* split LHS once so we can reuse for all passes */
            while ((next = subs.find(',', pos)) != std::string::npos)
            {
                lhs_subs = subs.substr(pos, next - pos);
                if (lhs_idx == i)
                    lhs_subs += "z";
                args.push_back(lhs_idx == i ? xs[i].d : xs[lhs_idx].r);
                pos = next + 1;
                ++lhs_idx;
            }
            /* last LHS term (until “->”) */
            lhs_subs = subs.substr(pos, subs.find("->") - pos);
            if (lhs_idx == i)
                lhs_subs += "z";
            args.push_back(lhs_idx == i ? xs[i].d : xs[lhs_idx].r);

            /* RHS */
            std::string rhs_subs = subs.substr(subs.find("->") + 2) + "z";
            std::string einsum_str = subs.substr(0, subs.find("->")) + "->" + rhs_subs;

            result_d += torch::einsum(einsum_str, args);
        }

        /*──────────────────────── accumulate second-order terms ───────────────*/
        for (size_t i = 0; i < xs.size(); ++i)
            for (size_t j = i + 1; j < xs.size(); ++j)
            {
                /* Build operand list with d_i and d_j, rest r_k */
                std::vector<torch::Tensor> args;
                args.reserve(xs.size());

                std::string lhs_subs;
                size_t pos = 0, next;
                size_t lhs_idx = 0;

                while ((next = subs.find(',', pos)) != std::string::npos)
                {
                    lhs_subs = subs.substr(pos, next - pos);
                    if (lhs_idx == i)
                        lhs_subs += "z";
                    if (lhs_idx == j)
                        lhs_subs += "w";
                    args.push_back(lhs_idx == i ? xs[i].d : lhs_idx == j ? xs[j].d
                                                                         : xs[lhs_idx].r);
                    pos = next + 1;
                    ++lhs_idx;
                }
                lhs_subs = subs.substr(pos, subs.find("->") - pos);
                if (lhs_idx == i)
                    lhs_subs += "z";
                if (lhs_idx == j)
                    lhs_subs += "w";
                args.push_back(lhs_idx == i ? xs[i].d : lhs_idx == j ? xs[j].d
                                                                     : xs[lhs_idx].r);

                std::string rhs_subs = subs.substr(subs.find("->") + 2) + "zw";
                std::string einsum_str = subs.substr(0, subs.find("->")) + "->" + rhs_subs;

                result_h += 2.0 * torch::einsum(einsum_str, args);
            }

        /*────────────────────────  h_i  terms  ────────────────────────────────*/
        for (size_t i = 0; i < xs.size(); ++i)
        {
            std::vector<torch::Tensor> args;
            args.reserve(xs.size());

            std::string lhs_subs;
            size_t pos = 0, next;
            size_t lhs_idx = 0;

            while ((next = subs.find(',', pos)) != std::string::npos)
            {
                lhs_subs = subs.substr(pos, next - pos);
                if (lhs_idx == i)
                    lhs_subs += "zw";
                args.push_back(lhs_idx == i ? xs[i].h : xs[lhs_idx].r);
                pos = next + 1;
                ++lhs_idx;
            }
            lhs_subs = subs.substr(pos, subs.find("->") - pos);
            if (lhs_idx == i)
                lhs_subs += "zw";
            args.push_back(lhs_idx == i ? xs[i].h : xs[lhs_idx].r);

            std::string rhs_subs = subs.substr(subs.find("->") + 2) + "zw";
            std::string einsum_str = subs.substr(0, subs.find("->")) + "->" + rhs_subs;

            result_h += torch::einsum(einsum_str, args);
        }

        return TensorHyperDual(result_r, result_d, result_h);
    }
};
