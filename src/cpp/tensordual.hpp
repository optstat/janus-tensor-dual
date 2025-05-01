#pragma once
#include <torch/torch.h>


#include <type_traits> // For std::is_scalar
#include <vector>
#include <sstream>  // For std::ostringstream
#include <iomanip>   // for std::setprecision
#include <stdexcept>
#include <variant>   // ← brings c10::get_if / holds_alternative
#include <c10/core/SymInt.h>   // for c10::SymInt::expect_int() etc.

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
    int64_t D_ = -1; // length of dual axis, set once in ctor
    torch::Dtype dtype;
    torch::Device device_ = torch::kCPU;

public:
    /// Invariant-checked ctor (no batch axis)
    TensorDual(const torch::Tensor& r_,
        const torch::Tensor& d_)
    {
        // Accept r_ : (N) or (1,N)        → squeeze to (N)
        TORCH_CHECK((r_.dim()==1) || (r_.dim()==2 && r_.size(0)==1),
                    "r must be 1-D (N) or (1,N); got ", r_.sizes());

        // Accept d_ : (N,D) or (1,N,D)    → squeeze to (N,D)
        TORCH_CHECK((d_.dim()==2 && r_.dim()==1) ||
                    (d_.dim()==3 && r_.dim()==2 && d_.size(0)==1),
                    "d must be 2-D (N,D) or 3-D (1,N,D); got ", d_.sizes());

        // Squeeze possible batch-1 axis
        r = (r_.dim()==2) ? r_.squeeze(0) : r_;
        d = (d_.dim()==3) ? d_.squeeze(0) : d;

        // Shape, dtype, device consistency
        TORCH_CHECK(r.size(0) == d.size(0),
                    "state axis mismatch: r N=", r.size(0),
                    " vs d N=", d.size(0));
        TORCH_CHECK(r.dtype()  == d.dtype()  &&
                    r.device() == d.device(),
                    "dtype / device mismatch");

        D_ = d.size(1);                       // derive D   (must be >0)
        TORCH_CHECK(D_ > 0,
                    "dual axis length D must be positive");
    }


    // move ctor/assign  – shallow moves are fine
    TensorDual(TensorDual&&) noexcept            = default;
    TensorDual& operator=(TensorDual&&) noexcept = default;
    // ─────────── Factory helpers ────────────────────────────────────────
    /// Create a TensorDual whose real part is all-zero (N)
    /// and whose dual part is all-zero (N,D).
    static TensorDual zeros(int64_t N,             // state dimension
                            int64_t D,             // dual-axis length
                            torch::Device dev = torch::kCPU,
                            torch::Dtype  dt  = torch::kFloat64)
    {
        auto opts = torch::TensorOptions().device(dev).dtype(dt);
        return TensorDual(torch::zeros({N},     opts),   // r  (N)
        torch::zeros({N, D},  opts));  // d  (N,D)
    }

    TensorMatDual eye() const;

    // ─────────── Element-wise operators (runtime-D safe) ────────────────
    friend TensorDual operator+(const TensorDual& a,
                                const TensorDual& b){
        TORCH_INTERNAL_ASSERT(a.D_ == b.D_ &&
         a.r.sizes() == b.r.sizes(),
         "shape/D mismatch in dual +");
        return { a.r + b.r,  a.d + b.d };
    }



    // Cheap, reference-counted copy
    TensorDual(const TensorDual&)            = default;
    TensorDual& operator=(const TensorDual&) = default;



    
    /// Return the complex conjugate (no-op for real tensors).
    inline TensorDual conj() const noexcept {
        // at::conj() works for both real and complex dtypes
        return TensorDual(r.conj(), d.conj());
    }

    /// Explicit deep copy (always duplicates storage).
    inline TensorDual deep_clone() const {
        return TensorDual(r.clone(), d.clone());
    }

    /// Pretty-print summary for debugging.
    ///  – prints shape / dtype / device and ‖·‖∞ for r and d.
    ///  – dumps full data only when the tensor has ≤ 20 elements.
    friend std::ostream& operator<<(std::ostream& os,
        const TensorDual& td)
    {
        auto dump = [&](const torch::Tensor& t,
        const char* label)
        {
            auto sz   = t.sizes();
            auto amax = t.abs().max().item<double>();

            os << "  " << label
            << "  shape=" << sz
            << "  dtype=" << t.dtype()
            << "  device="<< t.device()
            << "  |max|=" << amax;

            if (t.numel() <= 20) {
                os << "\n" << t << '\n';
            } else {
                os << "  (omitted " << t.numel() << " values)\n";
            }
        };

        os << "TensorDual @" << static_cast<const void*>(&td) << '\n';
        dump(td.r, "r");
        dump(td.d, "d");
        return os;
    }


    #ifdef TD_ENABLE_IO
    friend std::ostream& operator<<(std::ostream&, const TensorDual&);
    #endif
    
    /// Return a copy of this TensorDual on the requested device (dtype unchanged).
    inline TensorDual to(torch::Device dev,
        bool non_blocking = false) const {
        // r and d always share dtype/device, so one call per part
        auto r_new = r.to(dev, /*dtype=*/r.dtype(), non_blocking, /*copy=*/false);
        auto d_new = d.to(dev, /*dtype=*/d.dtype(), non_blocking, /*copy=*/false);
        return TensorDual(r_new, d_new);      // re-validates D_
    }

    /// Copy (or view) this TensorDual to a new device *and* dtype.
    /// Mirrors torch::Tensor::to(…).
    inline TensorDual to(torch::Device dev,
        torch::Dtype  dt,
        bool          non_blocking = false,
        bool          copy         = false) const
    {
        auto r_new = r.to(dev, dt, non_blocking, copy);  // device, dtype
        auto d_new = d.to(dev, dt, non_blocking, copy);
        return TensorDual(r_new, d_new);                 // re-checks invariant
    }


    /// Build a TensorDual whose dual part is an identity matrix
    /// (i.e. unit seed in every coordinate).  Only valid when D == N.
    static TensorDual eye(int64_t N,
        torch::Dtype  dt   = torch::kFloat64,
        torch::Device dev  = torch::kCPU)
    {
        auto opts = torch::TensorOptions().dtype(dt).device(dev);
        auto r = torch::zeros({N}, opts);          // or accept caller-supplied r
        auto d = torch::eye(N, opts);              // (N,N)  ⇒  D = N
        return TensorDual(r, d);                   // constructor re-validates
    }

    /// Return a TensorDual whose r and d are zeros with the
    /// same shape/dtype/device as the input.
    [[nodiscard]] inline static TensorDual zeros_like(const TensorDual& x) noexcept {
        return TensorDual(torch::zeros_like(x.r),
                        torch::zeros_like(x.d));   // constructor re-derives D_
    }



    /// Zeros shaped exactly like *this*
    [[nodiscard]] inline TensorDual zeros_like() const noexcept {
        return TensorDual(torch::zeros_like(r),
                        torch::zeros_like(d));   // ctor re-derives D_
    }

    /// Real part = 1, dual part = 0  (same shape / dtype / device as x)
    [[nodiscard]] inline static TensorDual ones_like(const TensorDual& x) noexcept {
        return TensorDual(torch::ones_like(x.r),      // (N)      ← all ones
                        torch::zeros_like(x.d));    // (N,D)    ← all zeros
    }


    /// Boolean tensor shaped like the real part of *x*
    /// (dtype = kBool, same device, same shape).
    [[nodiscard]] inline static torch::Tensor bool_like(const TensorDual& x) noexcept {
        return torch::zeros_like(x.r, x.r.options().dtype(torch::kBool));
    }


    /// Factory: real part = *r*, dual part = zeros, dual-axis length = ddim.
    ///   r    : shape (N)        or (1,N)   – any floating / complex dtype
    ///   ddim : D > 0
    [[nodiscard]] inline static TensorDual
    create_zero(const torch::Tensor& r, int64_t ddim)
    {
        TORCH_CHECK(r.defined(),   "create_zero: input tensor r is undefined");
        TORCH_CHECK(ddim > 0,      "create_zero: ddim must be > 0, got ", ddim);

        // Canonicalise r to shape (N)
        auto r_flat = (r.dim()==2) ? r.squeeze(0) : r;
        TORCH_CHECK(r_flat.dim()==1,
                    "create_zero: r must be 1-D or (1,N); got ", r.sizes());

        // Build dual tensor of shape (N, D)
        auto opts = r.options();
        auto d    = torch::zeros({r_flat.size(0), ddim}, opts);

        return TensorDual(r_flat, d);   // constructor re-derives D_
    }


    /// Uninitialised (garbage-filled) tensors with the same
    /// shape / dtype / device as *x*.
    /// Callers **must** write every element before reading!
    [[nodiscard]] inline static TensorDual empty_like(const TensorDual& x) noexcept {
        return TensorDual(torch::empty_like(x.r),     // (N)
                        torch::empty_like(x.d));    // (N,D)
    }

    /// Concatenate several TensorDual objects along axis 0 (state) or 1 (dual).
    /// All operands must share the other axis unchanged.
    [[nodiscard]] inline static TensorDual
    cat(const std::vector<TensorDual>& xs, int64_t dim /*0 or 1*/)
    {
        TORCH_CHECK(!xs.empty(), "TensorDual::cat: input vector is empty");
        TORCH_CHECK(dim==0 || dim==1,
                    "TensorDual::cat: dim must be 0 (state) or 1 (dual), got ", dim);

        const auto N = xs[0].r.size(0);
        const auto D = xs[0].d.size(1);
        const auto opts_r = xs[0].r.options();        // dtype & device to re-use
        const auto ref_dev  = xs[0].r.device();
        const auto ref_dtype= xs[0].r.dtype();

        /* shape / dtype / device parity check */
        for (const auto& t : xs) {
            TORCH_CHECK(t.r.device() == ref_dev && t.r.dtype() == ref_dtype,
                        "TensorDual::cat: dtype / device mismatch; expected ",
                        ref_dtype, " on ", ref_dev, " but got ",
                        t.r.dtype(), " on ", t.r.device());

            if (dim == 0) {                                // concat along state axis
                TORCH_CHECK(t.d.size(1) == D,
                            "TensorDual::cat dim=0: dual axis (D) mismatch, expected ",
                            D, " got ", t.d.size(1));
            } else {                                       // concat along dual axis
                TORCH_CHECK(t.r.size(0) == N,
                            "TensorDual::cat dim=1: state axis (N) mismatch, expected ",
                            N, " got ", t.r.size(0));
            }
        }


        // collect tensors
        std::vector<torch::Tensor> r_parts, d_parts;
        r_parts.reserve(xs.size());
        d_parts.reserve(xs.size());
        for (const auto& t : xs) {
            r_parts.push_back(t.r);
            d_parts.push_back(t.d);
        }

        auto r_cat = torch::cat(r_parts, /*dim=*/dim);
        auto d_cat = torch::cat(d_parts, /*dim=*/dim);

        return TensorDual(r_cat, d_cat);        // constructor re-validates (N,D)
    }

    /// Dual-aware einsum: apply `arg` to real parts and propagate duals
    ///
    ///  * `arg` **must** contain “->” in the usual PyTorch style, _without_ a “z”.
    ///  * The rule used is  d   =  r₁ ⊙ d₂  +  d₁ ⊙ r₂
    ///    (product rule, assuming commutative multiplication).
    ///
    ///  Example
    ///     tdC = TensorDual::einsum("bi,bj->bij", A, B);   // outer product
    ///
    [[nodiscard]] inline static TensorDual
    einsum(const std::string& arg,
           const TensorDual&  a,
           const TensorDual&  b) noexcept
    {
        // ── validation ────────────────────────────────────────────────────
        const auto comma = arg.find(',');
        const auto arrow = arg.find("->");
        TORCH_CHECK(comma != std::string::npos && arrow != std::string::npos,
                    "TensorDual::einsum: arg must contain ',' and '->', got ", arg);
        TORCH_CHECK(arg.find('z') == std::string::npos,                       // ★
                    "TensorDual::einsum: letter 'z' is reserved for dual axis");
        TORCH_CHECK(a.D_ == b.D_,
                    "TensorDual::einsum: dual-axis length mismatch (",
                    a.D_, " vs ", b.D_, ')');
        TORCH_CHECK(a.r.dtype()  == b.r.dtype()  &&
                    a.r.device() == b.r.device(),
                    "TensorDual::einsum: dtype/device mismatch between operands");
        
    
        // ── primal result ────────────────────────────────────────────────
        auto r_out = torch::einsum(arg, {a.r, b.r});
    
        // ── build dual subscripts (append 'z') ───────────────────────────
        const std::string lhs = arg.substr(0, comma);
        const std::string rhs = arg.substr(comma + 1, arrow - comma - 1);
        const std::string out = arg.substr(arrow + 2);
    
        const std::string sub_r_d2 = lhs + ","  + rhs + "z->" + out + "z";
        const std::string sub_d1_r = lhs + "z," + rhs +  "->" + out + "z";
    
        // ── dual result: r₁·d₂ + d₁·r₂ ──────────────────────────────────
        auto d_out = torch::einsum(sub_r_d2, {a.r, b.d}) +
                     torch::einsum(sub_d1_r, {a.d, b.r});
    
        return TensorDual(std::move(r_out), std::move(d_out));   // invariant re-checked
    }
    





    /// Einsum for (torch::Tensor  ×  TensorDual) → TensorDual
    ///
    ///   * `arg` must contain “,” and “->” and must **not** contain ‘z’
    ///   * Product rule:   d_out = einsum(first , second.d)
    ///
    /// Example:
    ///     td_out = TensorDual::einsum("ij,j->i",  A,  tdB);
    ///
    [[nodiscard]] inline static TensorDual
    einsum(const std::string& arg,
        const torch::Tensor& first,
        const TensorDual&    second) noexcept
    {
        // ── validate spec & operands ─────────────────────────────────────
        const auto comma = arg.find(',');
        const auto arrow = arg.find("->");
        TORCH_CHECK(comma != std::string::npos && arrow != std::string::npos,
                    "TensorDual::einsum(tensor,dual): arg must contain ',' and '->', got ", arg);
        TORCH_CHECK(arg.find('z') == std::string::npos,
                    "TensorDual::einsum(tensor,dual): letter 'z' is reserved for dual axis");

        // ── primal result ────────────────────────────────────────────────
        auto r_out = torch::einsum(arg, {first, second.r});

        // ── build dual sub-script  lhs ',' rhs+'z' -> out+'z' ───────────
        const std::string lhs = arg.substr(0, comma);
        const std::string rhs = arg.substr(comma + 1, arrow - comma - 1);
        const std::string out = arg.substr(arrow + 2);

        const std::string dual_sub = lhs + "," + rhs + "z->" + out + "z";

        // ── dual result ─────────────────────────────────────────────────
        auto d_out = torch::einsum(dual_sub, {first, second.d});

        return TensorDual(std::move(r_out), std::move(d_out));   // invariant re-checked
    }

    /// Dual-aware einsum : (TensorDual  ×  Tensor)  →  TensorDual
    ///
    ///   * `arg` must be binary and must NOT contain the letter ‘z’.
    ///   * Product rule:  d_out = einsum(lhs_z,rhs -> out_z)
    ///
    [[nodiscard]] inline static TensorDual
    einsum(const std::string& arg,
        const TensorDual&  first,
        const torch::Tensor& second) noexcept
    {
        // ── validate einsum string ──────────────────────────────────────────
        const auto comma = arg.find(',');
        const auto arrow = arg.find("->");
        TORCH_CHECK(comma != std::string::npos && arrow != std::string::npos && comma < arrow,
                    "TensorDual::einsum(dual,tensor): arg must contain ',' and '->'; got ", arg);
        TORCH_CHECK(arg.find('z') == std::string::npos,
                    "TensorDual::einsum(dual,tensor): letter 'z' is reserved for dual axis");
        
        TORCH_CHECK(first.r.device() == second.device(),
                    "einsum device mismatch: first on ", first.r.device(),
                    " second on ", second.device());
        

        // ── primal result ─────────────────────────────────────────────────
        auto r_out = torch::einsum(arg, {first.r, second});

        // ── build dual sub-script  lhs_z , rhs -> out_z ───────────────────
        const std::string lhs  = arg.substr(0, comma);
        const std::string rhs  = arg.substr(comma + 1, arrow - comma - 1);
        const std::string out  = arg.substr(arrow + 2);
        const std::string dual_sub = lhs + "z," + rhs + "->" + out + "z";

        // ── dual result ───────────────────────────────────────────────────
        auto d_out = torch::einsum(dual_sub, {first.d, second});

        return TensorDual(std::move(r_out), std::move(d_out));   // invariant re-checked
    }


    [[nodiscard]] inline static TensorDual
    einsum(const std::string& arg,
           const std::vector<TensorDual>& ts) noexcept
    {
        TORCH_CHECK(!ts.empty(), "TensorDual::einsum(vec): input vector is empty");
        TORCH_CHECK(arg.find("->") != std::string::npos &&
                    arg.find('z')  == std::string::npos,
                    "TensorDual::einsum(vec): arg must contain '->' and no 'z'");
    
        // Shape / dtype / device parity
        const int64_t D = ts[0].D_;
        const auto opts = ts[0].r.options();
        for (size_t i=0;i<ts.size();++i) {
            TORCH_CHECK(ts[i].D_ == D,
                        "dual-axis length mismatch at tensor ", i);
        }
    
        // Split positions of commas once
        std::vector<size_t> commas;
        for (size_t p = arg.find(','); p != std::string::npos; p = arg.find(',', p+1))
            commas.push_back(p);
    
        TORCH_CHECK(commas.size()+1 == ts.size(),
                    "TensorDual::einsum(vec): spec has ", commas.size()+1,
                    " operands but ", ts.size(), " tensors were provided");
    
        // Real parts → primal result
        std::vector<torch::Tensor> r_parts;
        r_parts.reserve(ts.size());
        for (const auto& t : ts) r_parts.push_back(t.r);
    
        auto r_out = torch::einsum(arg, r_parts);
    
        // Prepare for dual accumulation
        torch::Tensor d_out;                       // lazily initialised
        const size_t arrow = arg.find("->");
    
        // Loop over operands, replace kth real by kth dual
        for (size_t k = 0; k < ts.size(); ++k) {
            std::vector<torch::Tensor> ops = r_parts;
            ops[k] = ts[k].d;                      // (N,D)
    
            std::string darg = arg;                // copy
            darg.insert((k < commas.size() ? commas[k] : arrow), "z");
            darg.append("z");
    
            auto term = torch::einsum(darg, ops);
    
            if (k==0)
                d_out = term.clone();              // init
            else
                d_out += term;                     // accumulate
        }
    
        return TensorDual(std::move(r_out), std::move(d_out));   // invariant re-checked
    }
    

    /// Element-wise `torch::where` for TensorDual  (no batch axis)
    ///
    /// `cond` must be broadcast-compatible with `x.r` and boolean (`kBool`).
    [[nodiscard]] inline static TensorDual
    where(const torch::Tensor&  cond,
        const TensorDual&     x,
        const TensorDual&     y)
    {
        TORCH_CHECK(x.D_ == y.D_,
                    "`where`: dual-axis length mismatch (",
                    x.D_, " vs ", y.D_, ')');

        TORCH_CHECK(cond.dtype() == torch::kBool,
                    "`where`: condition tensor must be boolean, got ",
                    cond.dtype());


        // Real part – PyTorch broadcasts cond to (N) automatically
        auto xr = torch::where(cond, x.r, y.r);

        // Dual part – expand cond with one trailing axis for D
        auto cond_d = cond.unsqueeze(-1);               // shape (...,1)
        auto xd = torch::where(cond_d, x.d, y.d);       // broadcast over D

        return TensorDual(std::move(xr), std::move(xd));   // invariant re-checked
    }



    /// Sum over the state axis 0  (keeps dual axis intact).
    [[nodiscard]] inline static TensorDual sum(const TensorDual& x,
        bool keepdim = false) noexcept
    {
        auto r_sum = torch::sum(x.r, /*dim=*/0, /*keepdim=*/keepdim);      // ( ) or (1)
        auto d_sum = torch::sum(x.d, /*dim=*/0, /*keepdim=*/keepdim);      // (D) or (1,D)
        return TensorDual(r_sum, d_sum);                                   // re-checks (N,D)
    }


    /// Sum over the state axis (axis 0).  
    /// If keepdim == false the result is a length-1 state axis so the
    /// TensorDual invariant (same leading axis for r and d) is preserved.
    [[nodiscard]] inline TensorDual sum(bool keepdim = false) const noexcept
    {
        auto r_sum = torch::sum(r, /*dim=*/0, /*keepdim=*/keepdim);   // ()  or (1)
        auto d_sum = torch::sum(d, /*dim=*/0, /*keepdim=*/keepdim);   // (D) or (1,D)

        if (!keepdim) {                    // re-insert state axis = 1
            r_sum = r_sum.unsqueeze(0);    // (1)
            d_sum = d_sum.unsqueeze(0);    // (1,D)
        }
        return TensorDual(std::move(r_sum), std::move(d_sum));  // invariant re-checked
    }



    /// L2 norm of the real part  ‖r‖₂  (scalar)  and its dual propagation.
    /// Result shapes:  r_norm : (1)      d_norm : (1, D)
    [[nodiscard]] inline TensorDual normL2() const noexcept
    {
        // ── primal norm (scalar) ─────────────────────────────────────────
        auto r_norm = torch::norm(r, 2);            // shape ()

        // ── avoid divide-by-zero ───────────────────────────────────────
        auto r_norm_safe = r_norm.clamp_min(1e-12);

        // grad_r = r / ‖r‖₂          (shape (N))
        auto grad_r = r / r_norm_safe;

        // ── propagate to dual part  d_out_j = Σ_i grad_r_i * d_ij ──────
        auto d_out = torch::einsum("i,ij->j", {grad_r, d});   // shape (D)

        // ── re-insert state axis so shapes match constructor ───────────
        r_norm = r_norm.unsqueeze(0);   // (1)
        d_out  = d_out.unsqueeze(0);    // (1,D)

        return TensorDual(std::move(r_norm), std::move(d_out));   // invariant re-checked
    }


    /// Deep copy (duplicates tensor storage; mutations are independent).
    [[nodiscard]] inline TensorDual clone() const noexcept {
        return TensorDual(r.clone(),          // (N)
                        d.clone());         // (N,D)
    }


    /// Unary minus – negates real and dual parts.
    [[nodiscard]] inline TensorDual operator-() const noexcept
    {
        return TensorDual(-r, -d);          // constructor re-validates (N,D)
    }


    /// Element-wise addition of two TensorDuals
    [[nodiscard]] inline TensorDual operator+(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(D_ == other.D_,
                    "TensorDual::operator+: dual-axis length mismatch (",
                    D_, " vs ", other.D_, ')');
        TORCH_CHECK(r.sizes() == other.r.sizes(),
                    "TensorDual::operator+: state-axis length mismatch (",
                    r.sizes(), " vs ", other.r.sizes(), ')');

        return TensorDual(r + other.r,   // (N)
                        d + other.d);  // (N,D)
    }

    /// Add a plain tensor to the real part; dual part is untouched.
    ///
    /// * `other` must be broadcast-compatible with `r` **and** share
    ///   dtype / device.
    /// * Dual axis `(N,D)` remains unchanged (shallow copy is fine).
    [[nodiscard]] inline TensorDual operator+(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator+: RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator+: dtype / device mismatch");

        TORCH_CHECK(
            other.sizes() == r.sizes() ||                      // exact match
            other.dim() == 0,                                  // or scalar broadcast
            "TensorDual::operator+: shape incompatibility (r shape ", r.sizes(),
            " vs other shape ", other.sizes(), ')');

        return TensorDual(r + other, d);      // d is shared; no need to clone
    }

    /// Add a scalar to the real part; dual part unchanged.
    [[nodiscard]] inline TensorDual operator+(double other) const noexcept
    {
        TORCH_CHECK(r.is_floating_point() || r.is_complex(),
                    "TensorDual::operator+: scalar add requires floating/complex dtype");
        return TensorDual(r + other, d);   // `other` is implicitly turned into a scalar tensor
    }

    /// Element-wise subtraction of two TensorDuals.
    [[nodiscard]] inline TensorDual operator-(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(D_ == other.D_,
                    "TensorDual::operator-: dual-axis length mismatch (",
                    D_, " vs ", other.D_, ')');

        TORCH_CHECK(r.sizes() == other.r.sizes(),
                    "TensorDual::operator-: state-axis length mismatch (",
                    r.sizes(), " vs ", other.r.sizes(), ')');


        return TensorDual(r - other.r,          // (N)
                        d - other.d);         // (N,D)
    }


    /// Subtract a scalar from the real part; dual part unchanged.
    [[nodiscard]] inline TensorDual operator-(double scalar) const noexcept
    {
        TORCH_CHECK(r.is_floating_point() || r.is_complex(),
                    "TensorDual::operator-: scalar subtraction requires floating/complex dtype");

        return TensorDual(r - scalar, d);   // dual part shared; no clone needed
    }

    /// Return a TensorDual whose real and dual parts are contiguous
    /// in memory (no-op if they already are).
    [[nodiscard]] inline TensorDual contiguous() const noexcept
    {
        auto r_contig = r.is_contiguous() ? r : r.contiguous();
        auto d_contig = d.is_contiguous() ? d : d.contiguous();
        return TensorDual(std::move(r_contig), std::move(d_contig));
    }

    /// Element-wise multiplication of two TensorDuals
    ///
    ///   r_out = r ⊙ r_other
    ///   d_out = r_other ⊙ d + r ⊙ d_other     (product rule)
    [[nodiscard]] inline TensorDual operator*(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(D_ == other.D_,
                    "TensorDual::operator*: dual-axis length mismatch (",
                    D_, " vs ", other.D_, ')');
        TORCH_CHECK(r.sizes() == other.r.sizes(),
                    "TensorDual::operator*: state-axis length mismatch (",
                    r.sizes(), " vs ", other.r.sizes(), ')');

        // real part
        auto real = r * other.r;                                  // (N)

        // dual part  – broadcast scalar state axis to (N,1) then (N,D)
        auto dual = other.r.unsqueeze(-1) * d                     // r₂ ⊙ d₁
                + r.unsqueeze(-1)        * other.d;             // r₁ ⊙ d₂  → (N,D)

        return TensorDual(std::move(real), std::move(dual));      // invariant re-checked
    }


    /// Element-wise multiplication of a TensorDual by a plain tensor.
    /// The tensor scales **both** the real part and every dual slot.
    ///
    ///  * `other` must be either a scalar (0-D) or the same shape as `r` (N)
    ///    and share dtype / device with `r`.
    [[nodiscard]] inline TensorDual operator*(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator*(dual,tensor): RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator*(dual,tensor): dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator*(dual,tensor): RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        // real part
        auto real = r * other;                  // scalar or element-wise (N)

        // dual part – broadcast tensor over trailing dual axis (N,D)
        auto dual = d * other.unsqueeze(-1);    // (...,1) → broadcast on D

        return TensorDual(std::move(real), std::move(dual));   // invariant re-checked
    }

    /// Scale an entire TensorDual by a scalar.
    ///
    /// The scalar multiplies both the real part and every dual slot.
    [[nodiscard]] inline TensorDual operator*(double scalar) const noexcept
    {
        TORCH_CHECK(r.is_floating_point() || r.is_complex(),
                    "TensorDual::operator*: scalar multiplication requires "
                    "floating-point or complex dtype, got ", r.dtype());

        return TensorDual(r * scalar,          // (N)
                        d * scalar);         // (N,D)
    }

    /// Element-wise “<=” comparison on **real parts**.
    /// Returns a boolean tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator<=(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::operator<=: shape mismatch (", r.sizes(),
                    " vs ", other.r.sizes(), ')');

        return r <= other.r;        // (N) bool
    }


    /// Element-wise equality on the real parts of two TensorDuals.  
    /// Dual parts are intentionally ignored.
    [[nodiscard]] inline torch::Tensor operator==(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::operator==: shape mismatch (", r.sizes(),
                    " vs ", other.r.sizes(), ')');

        return r == other.r;           // bool tensor shape (N)
    }




    /// Element-wise “<=” comparison between the real part of a TensorDual
    /// and a plain tensor (scalar or same shape).  
    /// Dual part is ignored.
    [[nodiscard]] inline torch::Tensor operator<=(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator<=: RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator<=: dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator<=: RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r <= other;          // bool tensor, shape (N) or broadcast result
    }

    /// Compare the real part with a scalar (element-wise “<=”).
    /// Dual part is ignored.  Result shape = (N) bool tensor.
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator<=(Scalar scalar) const noexcept
    {
        // PyTorch automatically converts the C++ scalar to the tensor’s dtype and broadcasts it.
        return r <= scalar;        // bool tensor, shape (N)
    }


    /// Element-wise “greater-than” comparison on real parts.
    /// Dual components are intentionally ignored.
    [[nodiscard]] inline torch::Tensor operator>(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::operator>: shape mismatch (", r.sizes(),
                    " vs ", other.r.sizes(), ')');

        return r > other.r;           // bool tensor, shape (N)
    }


    /// Element-wise “greater-than” comparison between the real part of a
    /// TensorDual and a plain tensor (scalar or same shape).  
    /// Dual part is ignored.  Result is a bool tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator>(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator>: RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator>: dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator>: RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r > other;          // bool tensor  (N)
    }

    /// Element-wise “greater-than” comparison between the real part of a
    /// TensorDual and a scalar.  Dual part is ignored.
    ///
    /// Returns a boolean tensor of shape (N).
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator>(Scalar scalar) const noexcept
    {
        return r > scalar;           // bool tensor, shape (N)
    }



    /// Element-wise “less-than” comparison between the real part of a
    /// TensorDual and a plain tensor (scalar or same shape).  
    /// Dual part is ignored.
    [[nodiscard]] inline torch::Tensor operator<(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator<: RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator<: dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator<: RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r < other;           // bool tensor, shape (N)
    }

    /// Element-wise “less-than” comparison between the real part of a
    /// TensorDual and a scalar value.  
    /// Dual slots are ignored.  Result: bool tensor of shape (N).
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator<(Scalar scalar) const noexcept
    {
        return r < scalar;      // broadcasts C++ scalar, returns (N) bool
    }


    /// Element-wise “greater-than-or-equal” comparison on the real parts of two
    /// TensorDuals.  Dual components are ignored.  
    /// Result is a boolean tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator>=(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::operator>=: shape mismatch (", r.sizes(),
                    " vs ", other.r.sizes(), ')');

        return r >= other.r;          // bool tensor, shape (N)
    }

    /// Element-wise “≥” comparison between the real part of a TensorDual
    /// and a plain tensor (scalar or same shape).  
    /// Dual slots are ignored; result is a bool tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator>=(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator>= : RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator>= : dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator>= : RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r >= other;                     // bool tensor, shape (N)
    }


    /// Element-wise “≥” comparison between the real part of a TensorDual and
    /// a numeric scalar.  Dual slots are ignored.  Result: bool tensor (N).
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator>=(Scalar scalar) const noexcept
    {
        return r >= scalar;        // broadcasts scalar, returns shape (N)
    }


    /// Element-wise equality comparison between the real part of a TensorDual
    /// and a plain tensor (scalar × broadcast or same shape).  
    /// Dual part is ignored.  Result: bool tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator==(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator== : RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator== : dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator== : RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r == other;              // bool tensor, shape (N)
    }

    /// Element-wise equality between the real part of a TensorDual and a numeric
    /// scalar.  Dual slots are ignored.  Returns a bool tensor of shape (N).
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator==(Scalar scalar) const noexcept
    {
        return r == scalar;      // broadcasts C++ scalar, result (N) bool
    }

    /// Element-wise inequality comparison on the real parts of two TensorDuals.
    /// Dual components are ignored.  Result is a bool tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator!=(const TensorDual& other) const noexcept
    {
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::operator!= : shape mismatch (", r.sizes(),
                    " vs ", other.r.sizes(), ')');

        return r != other.r;     // bool tensor (N)
    }

    /// Element-wise “!=” comparison between the real part of a TensorDual and a
    /// plain tensor (scalar or same shape).  
    /// Dual slots are ignored.  Result is a bool tensor of shape (N).
    [[nodiscard]] inline torch::Tensor operator!=(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator!= : RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator!= : dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator!= : RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        return r != other;                 // bool tensor, shape (N)
    }



    /// Element-wise “!=” comparison between the real part of a TensorDual and
    /// a numeric scalar.  Dual slots are ignored.  Result: bool tensor (N).
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    [[nodiscard]] inline torch::Tensor operator!=(Scalar scalar) const noexcept
    {
        return r != scalar;          // broadcasts scalar, returns shape (N)
    }

    /// Element-wise division of two TensorDuals  
    /// r_out = r₁ / r₂  
    /// d_out = (d₁ · r₂ − r₁ · d₂) / r₂²     (quotient rule, broadcast over D)
    [[nodiscard]] inline TensorDual operator/(const TensorDual& other) const noexcept
    {
        // ── compatibility checks ─────────────────────────────────────────
        TORCH_CHECK(D_ == other.D_,
                    "TensorDual::operator/: dual-axis length mismatch (",
                    D_, " vs ", other.D_, ')');
        TORCH_CHECK(r.sizes() == other.r.sizes(),
                    "TensorDual::operator/: state-axis length mismatch (",
                    r.sizes(), " vs ", other.r.sizes(), ')');

        // ── safeguard denominator (avoid divide-by-zero) ────────────────
        auto safe_r2 = torch::sign(other.r) * other.r.abs().clamp_min(1e-12); // (N)

        // ── quotient rule ───────────────────────────────────────────────
        auto real = r / safe_r2;                            // (N)

        auto denom_sq = safe_r2.square();                   // (N)
        auto dual = ( d / safe_r2.unsqueeze(-1)             //  d₁ / r₂
                    - (r / denom_sq).unsqueeze(-1) * other.d ); // − r₁ d₂ / r₂²

        return TensorDual(std::move(real), std::move(dual));   // invariant re-checked
    }

    /// Element-wise division of a TensorDual by a plain tensor `other`.
    /// Dual part follows the quotient-rule simplification  
    ///      d_out = d / other                    (since other has no dual)
    ///
    /// `other` may be:
    ///   • a scalar (0-D) – broadcast to every element of `r`  
    ///   • a vector of shape `(N)` identical to `r`  
    ///
    /// Dual axis **D** remains unchanged.
    [[nodiscard]] inline TensorDual operator/(const torch::Tensor& other) const noexcept
    {
        TORCH_CHECK(other.defined(),
                    "TensorDual::operator/(dual,tensor): RHS tensor is undefined");

        TORCH_CHECK(other.device() == r.device() &&
                    other.dtype()  == r.dtype(),
                    "TensorDual::operator/(dual,tensor): dtype/device mismatch");

        TORCH_CHECK(other.dim() == 0 || other.sizes() == r.sizes(),
                    "TensorDual::operator/(dual,tensor): RHS shape ",
                    other.sizes(), " is not broadcast-compatible with real part ",
                    r.sizes());

        // ---- make denominator safe (avoid divide-by-zero) -----------------
        auto safe_den = torch::sign(other) * other.abs().clamp_min(1e-12);   // 0-D or (N)

        // ---- real part -----------------------------------------------------
        auto real = r / safe_den;                         // broadcasts automatically

        // ---- dual part  d / denom  (broadcast over trailing D) ------------
        auto dual = d / safe_den.unsqueeze(-1);           // (...,1) → (N,D)

        return TensorDual(std::move(real), std::move(dual));   // invariant re-checked
    }



    TensorMatDual operator/(const TensorMatDual& other) const; // Forward declaration

    /// Divide a TensorDual by a scalar (element-wise).
    /// Dual slots scale by the same factor.
    ///
    ///     r_out = r / scalar
    ///     d_out = d / scalar
    ///
    /// A tiny clamp prevents division-by-zero while preserving sign.
    [[nodiscard]] inline TensorDual operator/(double scalar) const noexcept
    {
        TORCH_CHECK(r.is_floating_point() || r.is_complex(),
                    "TensorDual::operator/: scalar division requires floating/complex dtype");

        /* avoid 0-div; keep sign */
        double safe = std::copysign(std::max(std::abs(scalar), 1e-12), scalar);

        return TensorDual(r / safe,      // (N)
                        d / safe);     // (N,D)
    }


    /// Gather by state indices on axis 0.
    /// `index` must be an int/long tensor;  result shapes:
    ///     r_out : (K)
    ///     d_out : (K, D)
    [[nodiscard]] inline TensorDual gather(const torch::Tensor& index) const noexcept
    {
        TORCH_CHECK(index.dtype()  == torch::kLong,
                    "TensorDual::gather: index tensor must be int64");
        TORCH_CHECK(index.device() == r.device(),
                    "TensorDual::gather: index tensor must be on the same device");

        // real part: (N) -> (K)
        auto r_gather = r.index_select(/*dim=*/0, index);

        // dual part: broadcast index over the trailing dual axis D
        auto d_gather = d.index_select(/*dim=*/0, index);   // (K,D)

        return TensorDual(std::move(r_gather), std::move(d_gather));
    }


    /// Scatter `src` into *this* at the given **state indices** (axis 0).
    /// `index` must be an int64 tensor on the same device.
    /// Shapes:
    ///   • r_out : (N)
    ///   • d_out : (N,D)
    [[nodiscard]] inline TensorDual
    scatter(const torch::Tensor& index,
            const TensorDual&    src) const noexcept
    {
        // ── basic checks ────────────────────────────────────────────────
        TORCH_CHECK(index.dtype() == torch::kLong,
                    "TensorDual::scatter: index tensor must be int64");
        TORCH_CHECK(index.device() == r.device(),
                    "TensorDual::scatter: index tensor is on ", index.device(),
                    " but data are on ", r.device());

        TORCH_CHECK(src.D_ == D_ &&
                    src.r.sizes() == r.sizes(),
                    "TensorDual::scatter: src shape/D mismatch");

        // ── scatter real part (axis 0) ──────────────────────────────────
        auto r_out = r.clone();                         // keep original tensor
        r_out.index_put_({index}, src.r.index({index}));

        // ── scatter dual part (axis 0, broadcast over D) ───────────────
        auto d_out = d.clone();
        d_out.index_put_({index, /*all D*/ torch::indexing::Slice()},
                        src.d.index({index, torch::indexing::Slice()}));

        return TensorDual(std::move(r_out), std::move(d_out));   // invariant re-checked
    }

    /// Reciprocal of a TensorDual.
    ///   y = 1 / x
    ///   dy = − dx / x²
    ///
    /// A tiny `eps` is used to prevent division-by-zero while preserving sign.
    [[nodiscard]] inline TensorDual reciprocal(double eps = 1e-12) const noexcept
    {
        // ── protect zeros, keep sign ───────────────────────────────────
        auto safe_r = torch::where(r.abs() < eps,
                                torch::sign(r) * eps,
                                r);                 // (N)

        // ── primal  y = 1 / x ─────────────────────────────────────────
        auto inv_r = safe_r.reciprocal();              // (N)

        // ── dual    dy = −dx / x²  (broadcast over D) ─────────────────
        auto inv_sq = inv_r * inv_r;                   // (N)
        auto dual   = -d * inv_sq.unsqueeze(-1);       // (N,D)

        return TensorDual(std::move(inv_r), std::move(dual));   // invariant re-checked
    }

    /// Square a TensorDual.  
    ///   y  = x²  
    ///   dy = 2·x·dx
    [[nodiscard]] inline TensorDual square() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::square: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        // real part: x²
        auto rsq = r.square();                              // (N)

        // dual part: 2·x·dx   (broadcast x over the dual axis D)
        auto dual = 2.0 * r.unsqueeze(-1) * d;              // (N,D)

        return TensorDual(std::move(rsq), std::move(dual)); // invariant re-checked
    }

    /// Element-wise sine of a TensorDual.  
    ///   y  = sin(x)  
    ///   dy = cos(x) · dx
    [[nodiscard]] inline TensorDual sin() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::sin: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        // Primal
        auto r_sin = torch::sin(r);                    // (N)

        // Derivative:  cos(x) · dx
        auto dual   = torch::cos(r).unsqueeze(-1) * d; // (N,D)

        return TensorDual(std::move(r_sin), std::move(dual));   // invariant re-checked
    }

    /// Element-wise cosine of a TensorDual.  
    ///   y  = cos(x)  
    ///   dy = −sin(x) · dx
    [[nodiscard]] inline TensorDual cos() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::cos: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        // primal          cos(x)
        auto r_cos  = torch::cos(r);                     // (N)

        // derivative    −sin(x) · dx
        auto dual    = (-torch::sin(r)).unsqueeze(-1) * d;   // (N,D)

        return TensorDual(std::move(r_cos), std::move(dual));   // invariant re-checked
    }


    /// Element-wise tangent of a TensorDual.  
    ///   y  = tan(x)  
    ///   dy = sec²(x) · dx   with  sec²(x) = 1 / cos²(x)
    [[nodiscard]] inline TensorDual tan() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::tan: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        // ── primal  tan(x) ──────────────────────────────────────────────
        auto r_tan = torch::tan(r);                                 // (N)

        // ── derivative  sec²(x) · dx  (broadcast over D) ──────────────
        auto sec2  = 1.0 / torch::cos(r).square();                  // (N)
        auto dual  = sec2.unsqueeze(-1) * d;                        // (N,D)

        return TensorDual(std::move(r_tan), std::move(dual));       // invariant re-checked
    }

    /// Element-wise arcsine of a TensorDual.  
    ///   y  = asin(x)  
    ///   dy = dx / √(1 − x²)
    ///
    /// Domain is limited to |x| ≤ 1; near ±1 we clamp the denominator
    /// to avoid NaNs while preserving sign.
    [[nodiscard]] inline TensorDual asin(double eps = 1e-12) const noexcept
    {
        // ── domain check (debug builds) ──────────────────────────────────
        TORCH_CHECK((r.abs() <= 1 + 1e-14).all().item<bool>(),
                    "TensorDual::asin: real part has values outside [-1,1]");

        // ── primal  asin(x) ─────────────────────────────────────────────
        auto r_asin = torch::asin(r);                       // (N)

        // ── derivative  dx / √(1 − x²)  (broadcast over D) ─────────────
        auto denom = (1.0 - r.square()).clamp_min(eps).sqrt();   // (N)
        auto dual  = d / denom.unsqueeze(-1);                    // (N,D)

        return TensorDual(std::move(r_asin), std::move(dual));   // invariant re-checked
    }

    /// Element-wise arccosine of a TensorDual.  
    ///   y  = acos(x)  
    ///   dy = −dx / √(1 − x²)
    ///
    /// Domain: |x| ≤ 1.  A tiny clamp keeps the denominator finite at |x| → 1.
    [[nodiscard]] inline TensorDual acos(double eps = 1e-12) const noexcept
    {
        // ── domain check (debug builds) ──────────────────────────────────
        TORCH_CHECK((r.abs() <= 1 + 1e-14).all().item<bool>(),
                    "TensorDual::acos: real part has values outside [-1,1]");

        // ── primal  acos(x) ─────────────────────────────────────────────
        auto r_acos = torch::acos(r);                       // (N)

        // ── derivative  −dx / √(1 − x²)  (broadcast over D) ────────────
        auto denom = (1.0 - r.square()).clamp_min(eps).sqrt();  // (N)
        auto dual  = -d / denom.unsqueeze(-1);                 // (N,D)

        return TensorDual(std::move(r_acos), std::move(dual));  // invariant re-checked
    }


    /// Element-wise hyperbolic sine of a TensorDual.  
    ///   y  = sinh(x)  
    ///   dy = cosh(x) · dx
    [[nodiscard]] inline TensorDual sinh() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::sinh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        // primal
        auto r_sinh = torch::sinh(r);                     // (N)

        // derivative  cosh(x) · dx  (broadcast across D)
        auto dual    = torch::cosh(r).unsqueeze(-1) * d;  // (N,D)

        return TensorDual(std::move(r_sinh), std::move(dual));   // invariant re-checked
    }


    /// Element-wise hyperbolic cosine of a TensorDual.  
    ///   y  = cosh(x)  
    ///   dy = sinh(x) · dx
    [[nodiscard]] inline TensorDual cosh() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::cosh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal part */
        auto r_cosh = torch::cosh(r);                     // (N)

        /* derivative  sinh(x) · dx   — broadcast over trailing dual axis D */
        auto dual    = torch::sinh(r).unsqueeze(-1) * d;  // (N,D)

        return TensorDual(std::move(r_cosh), std::move(dual));   // invariant re-checked
    }

    /// Element-wise hyperbolic tangent of a TensorDual.  
    ///   y  = tanh(x)  
    ///   dy = (1 − tanh²(x)) · dx
    [[nodiscard]] inline TensorDual tanh() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::tanh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal */
        auto r_tanh = torch::tanh(r);                        // (N)

        /* derivative  (1 - tanh²(x)) · dx  — broadcast over D */
        auto factor = 1.0 - r_tanh.square();                 // (N)
        auto dual   = factor.unsqueeze(-1) * d;              // (N,D)

        return TensorDual(std::move(r_tanh), std::move(dual));   // invariant re-checked
    }

    /// Element-wise hyperbolic arcsine of a TensorDual.  
    ///   y  = asinh(x)  
    ///   dy = dx / √(1 + x²)
    [[nodiscard]] inline TensorDual asinh() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::asinh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal ---------------------------------------------------------------- */
        auto r_asinh = torch::asinh(r);                       // (N)

        /* derivative  dx / √(1 + x²)  (broadcast across dual axis D) ------------ */
        auto factor  = (1.0 + r.square()).sqrt().reciprocal();   // (N)
        auto dual    = factor.unsqueeze(-1) * d;                 // (N,D)

        return TensorDual(std::move(r_asinh), std::move(dual));  // invariant re-checked
    }



    /// Element-wise hyperbolic arccosine of a TensorDual.  
    ///   y  = acosh(x)     (valid for x ≥ 1)  
    ///   dy = dx / √(x² − 1)
    [[nodiscard]] inline TensorDual acosh(double eps = 1e-12) const noexcept
    {
        /* domain check ---------------------------------------------------------- */
        TORCH_CHECK((r >= 1.0 - 1e-14).all().item<bool>(),
                    "TensorDual::acosh: real part has elements < 1");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::acosh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal ---------------------------------------------------------------- */
        auto r_acosh = torch::acosh(r);                       // (N)

        /* derivative  dx / √(x² − 1)  (broadcast over D) ------------------------ */
        auto denom = (r.square() - 1.0).clamp_min(eps).sqrt(); // (N)  protect x≈1
        auto dual  = d / denom.unsqueeze(-1);                  // (N,D)

        return TensorDual(std::move(r_acosh), std::move(dual));   // invariant re-checked
    }

    /// Element-wise hyperbolic arctangent of a TensorDual.  
    ///   y  = atanh(x)              (valid for |x| < 1)  
    ///   dy = dx / (1 − x²)
    [[nodiscard]] inline TensorDual atanh(double eps = 1e-12) const noexcept
    {
        /* ── domain guard ─────────────────────────────────────────────── */
        TORCH_CHECK((r.abs() < 1.0 + 1e-14).all().item<bool>(),
                    "TensorDual::atanh: real part has elements outside (-1,1)");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::atanh: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* ── primal  atanh(x) ─────────────────────────────────────────── */
        auto r_atanh = torch::atanh(r);                       // (N)

        /* ── derivative  dx / (1 − x²)  (broadcast over D) ───────────── */
        auto denom = (1.0 - r.square()).clamp_min(eps);       // (N)   protect |x|→1
        auto dual  = d / denom.unsqueeze(-1);                 // (N,D)

        return TensorDual(std::move(r_atanh), std::move(dual));   // invariant re-checked
    }


    /// Element-wise exponential of a TensorDual.  
    ///   y  = exp(x)  
    ///   dy = exp(x) · dx
    [[nodiscard]] inline TensorDual exp() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::exp: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal */
        auto r_exp = torch::exp(r);                    // (N)

        /* derivative  exp(x) · dx  – broadcast over trailing dual axis D */
        auto dual   = r_exp.unsqueeze(-1) * d;         // (N,D)

        return TensorDual(std::move(r_exp), std::move(dual));   // invariant re-checked
    }


    /// Natural logarithm of a TensorDual.  
    ///   y  = log(x)   (valid for x > 0)  
    ///   dy = dx / x
    [[nodiscard]] inline TensorDual log(double eps = 1e-12) const noexcept
    {
        /* ── domain check ─────────────────────────────────────────────── */
        TORCH_CHECK((r > 0).all().item<bool>(),
                    "TensorDual::log: real part contains non-positive values");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::log: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* ── primal  log(x) ───────────────────────────────────────────── */
        auto r_log = torch::log(r);                       // (N)

        /* ── derivative  dx / x  (broadcast over dual axis D) ────────── */
        auto inv_r = (r + eps).reciprocal();              // (N)   clamp to avoid 0-div
        auto dual  = inv_r.unsqueeze(-1) * d;             // (N,D)

        return TensorDual(std::move(r_log), std::move(dual));   // invariant re-checked
    }

    /// Element-wise square-root of a TensorDual.  
    ///   y  = √x  
    ///   dy = dx / (2√x)
    ///
    /// * For x ≥ 0 it stays real.
    /// * If any x < 0 **and** `r` is not complex already, the result is promoted
    ///   to complex<dtype> (both r and d).
    [[nodiscard]] inline TensorDual sqrt() const
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::sqrt: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        const bool has_negative = (!torch::is_complex(r) &&
                                !(r >= 0).all().item<bool>());

        torch::Tensor r_sqrt, d_scaled;

        if (has_negative) {
            /* promote to complex ───────────────────────────────────────── */
            auto r_c = r.to(torch::toComplexType(r.scalar_type()));
            r_sqrt   = torch::sqrt(r_c);                       // (N) complex

            auto factor = 0.5 / r_sqrt;                        // (N) complex
            auto d_c    = d.to(torch::toComplexType(d.scalar_type()));
            d_scaled    = factor.unsqueeze(-1) * d_c;          // (N,D) complex
        } else {
            /* standard real branch ─────────────────────────────────────── */
            r_sqrt = torch::sqrt(r);                           // (N) real
            auto factor = 0.5 / r_sqrt;                        // (N)
            d_scaled = factor.unsqueeze(-1) * d;               // (N,D)
        }

        return TensorDual(std::move(r_sqrt), std::move(d_scaled)); // invariant re-checked
    }



    /// Element-wise absolute value of a TensorDual (real input).  
    ///   y  = |x|  
    ///   dy = sign(x) · dx
    ///
    /// * For complex inputs the derivative of |z| is not analytic; we
    ///   therefore reject complex `r` here.
    [[nodiscard]] inline TensorDual abs() const
    {
        TORCH_CHECK(!torch::is_complex(r),
                    "TensorDual::abs: derivative of complex magnitude "
                    "is not implemented (non-analytic).");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::abs: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* primal ----------------------------------------------------------- */
        auto r_abs = torch::abs(r);                    // (N)

        /* derivative  sign(x) · dx  (broadcast over dual axis D) ----------- */
        auto sign   = torch::sgn(r).unsqueeze(-1);     // (N,1)  sign(0)=0
        auto dual   = sign * d;                        // (N,D)

        return TensorDual(std::move(r_abs), std::move(dual));   // invariant re-checked
    }

    /// Element-wise signum of a TensorDual.  
    ///   y  = sign(x)         (−1, 0, +1 for x < 0, =0, >0)  
    ///   dy = 0               (derivative is zero a.e.)
    ///
    /// Only defined for real inputs; complex magnitude’s sign is ambiguous.
    [[nodiscard]] inline TensorDual sign() const noexcept
    {
        TORCH_CHECK(!torch::is_complex(r),
                    "TensorDual::sign: complex inputs are not supported");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::sign: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        auto r_sign = torch::sgn(r);            // (N)   −1 / 0 / +1
        auto d_zero = torch::zeros_like(d);     // (N,D) derivative ≡ 0

        return TensorDual(std::move(r_sign), std::move(d_zero));  // invariant re-checked
    }

    /// Signed logarithm  
    ///   f(x)  = sign(x) · log(|x| + 1)       (undefined at x = 0)  
    ///   f'(x) = 1 / (|x| + 1)                (signs cancel)
    [[nodiscard]] inline TensorDual slog() const
    {
        /* ── validity checks ─────────────────────────────────────────── */
        TORCH_CHECK(!torch::is_complex(r),
                    "TensorDual::slog: complex inputs are not supported");

        TORCH_CHECK((r != 0).all().item<bool>(),
                    "TensorDual::slog: argument contains zero, slog undefined there");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::slog: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* ── primal  sign(x)·log(|x|+1) ─────────────────────────────── */
        auto abs_r   = torch::abs(r);                      // (N)
        auto r_slog  = torch::sgn(r) * torch::log(abs_r + 1.0);

        /* ── derivative  dx / (|x|+1)  (broadcast over D) ───────────── */
        auto factor  = (abs_r + 1.0).reciprocal();         // (N)
        auto dual    = factor.unsqueeze(-1) * d;           // (N,D)

        return TensorDual(std::move(r_slog), std::move(dual));   // invariant re-checked
    }

    /// Inverse signed logarithm  
    ///   f(x)  = sign(x)·(exp(|x|) − 1)  
    ///   f'(x) = exp(|x|)
    [[nodiscard]] inline TensorDual sloginv() const noexcept
    {
        /* -------- sanity checks ----------------------------------------- */
        TORCH_CHECK(!torch::is_complex(r),
                    "TensorDual::sloginv: complex inputs are not supported");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::sloginv: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* -------- primal ------------------------------------------------ */
        auto abs_r     = torch::abs(r);                  // (N)
        auto exp_abs_r = torch::exp(abs_r);              // (N)
        auto r_out     = torch::sgn(r) * (exp_abs_r - 1.0);

        /* -------- derivative  exp(|x|)·dx  (broadcast over D) ----------- */
        auto dual_out  = exp_abs_r.unsqueeze(-1) * d;    // (N,D)

        return TensorDual(std::move(r_out), std::move(dual_out));   // invariant re-checked
    }

    /// Soft-sign
    ///   y  = x / (1 + |x|)  
    ///   dy = dx / (1 + |x|)²
    [[nodiscard]] inline TensorDual softsign() const noexcept
    {
        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::softsign: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* ── primal ─────────────────────────────────────────────────── */
        auto denom   = 1.0 + torch::abs(r);          // (N)
        auto r_out   = r / denom;                    // (N)

        /* ── derivative  dx / (1+|x|)²  (broadcast over D) ─────────── */
        auto factor  = denom.pow(-2);                // (N)
        auto dual_out= factor.unsqueeze(-1) * d;     // (N,D)

        return TensorDual(std::move(r_out), std::move(dual_out));  // invariant re-checked
    }




    /// Inverse soft-sign  
    ///   y  = x / (1 − |x|)  (valid for |x| < 1)  
    ///   dy = dx / (1 − |x|)²
    [[nodiscard]] inline TensorDual softsigninv() const noexcept
    {
        /* ---------- domain check ------------------------------------------------ */
        TORCH_CHECK((torch::abs(r) < 1.0).all().item<bool>(),
                    "TensorDual::softsigninv: |x| must be < 1 for every element");

        TORCH_CHECK(d.size(0) == r.size(0),
                    "TensorDual::softsigninv: state-axis length mismatch (",
                    d.size(0), " vs ", r.size(0), ')');

        /* ---------- primal  y = x / (1 − |x|) ----------------------------------- */
        auto abs_r   = torch::abs(r);                  // (N)
        auto denom   = 1.0 - abs_r;                    // (N)  positive by domain
        auto r_out   = r / denom;                      // (N)

        /* ---------- derivative  dx / (1−|x|)²  (broadcast over D) -------------- */
        auto factor  = denom.pow(-2);                  // (N)
        auto dual    = factor.unsqueeze(-1) * d;       // (N,D)

        return TensorDual(std::move(r_out), std::move(dual));   // invariant re-checked
    }

    /// max() over the state axis (axis 0)
    /// Result shapes:
    ///    r_max : (1)      – keep a length-1 state axis
    ///    d_max : (1, D)   – dual slot that belongs to that max
    [[nodiscard]] inline TensorDual max() const noexcept
    {
        // single state axis → dim = 0
        auto max_pair  = torch::max(r, /*dim=*/0, /*keepdim=*/false);
        auto r_max     = std::get<0>(max_pair);            // shape ()
        auto idx_max   = std::get<1>(max_pair);            // shape ()

        // keep state axis length = 1 to preserve (N)/(N,D) invariant
        r_max = r_max.unsqueeze(0);                        // (1)

        // pick the same row from the dual tensor
        auto d_max = d.index_select(/*dim=*/0,
                                    idx_max.unsqueeze(0)); // (1,D)

        return TensorDual(std::move(r_max), std::move(d_max)); // invariant re-checked
    }

    /// Min over the state axis (axis 0).  
    /// Returns the minimal real value together with its matching dual slot.
    ///
    /// Shapes of the result respect the invariant:
    ///   r_min : (1)      – keeps a length-1 state axis  
    ///   d_min : (1, D)
    [[nodiscard]] inline TensorDual min() const noexcept
    {
        // single state axis → dim = 0
        auto min_pair = torch::min(r, /*dim=*/0, /*keepdim=*/false);
        auto r_min    = std::get<0>(min_pair);            // shape ()
        auto idx_min  = std::get<1>(min_pair);            // shape ()

        // reinstate state axis length = 1
        r_min = r_min.unsqueeze(0);                       // (1)

        // pick the corresponding row from the dual tensor
        auto d_min = d.index_select(/*dim=*/0,
                                    idx_min.unsqueeze(0)); // (1, D)

        return TensorDual(std::move(r_min), std::move(d_min));   // invariant re-checked
    }

    /// Promote real & dual parts to matching complex dtype (if not already).
    [[nodiscard]] inline TensorDual complex() const noexcept
    {
        auto target_cdtype = torch::toComplexType(r.scalar_type());  // keeps precision

        auto r_c = r.is_complex()
                ? r
                : torch::complex(r, torch::zeros_like(r)).to(target_cdtype);

        auto d_c = d.is_complex()
                ? d
                : torch::complex(d, torch::zeros_like(d)).to(target_cdtype);

        return TensorDual(std::move(r_c), std::move(d_c));   // shapes (N) / (N,D)
    }

    /// Extract the real component of both r and d.
    /// If they are already real, this is a cheap no-op view.
    [[nodiscard]] inline TensorDual real() const noexcept
    {
        auto r_real = r.is_complex() ? torch::real(r) : r;
        auto d_real = d.is_complex() ? torch::real(d) : d;
        return TensorDual(r_real, d_real);   // shapes unchanged
    }

    /// Imaginary component of r and d.
    /// For real tensors this is automatically zeros_like(x).
    [[nodiscard]] inline TensorDual imag() const noexcept
    {
        auto imag_r = torch::imag(r);   // (N)
        auto imag_d = torch::imag(d);   // (N,D)

        return TensorDual(std::move(imag_r), std::move(imag_d));  // invariant re-checked
    }
    
    
    /// Select rows by advanced indexing (state axis 0 only).
    /// `index` must be a 1-D int64 tensor on the same device.
    /// Shapes of the result:
    ///     r_out : (K)      → unsqueezed to (1,K)  ➜ (K) after constructor
    ///     d_out : (K, D)
    [[nodiscard]] inline TensorDual index_select(const torch::Tensor& index) const
    {
        TORCH_CHECK(index.dtype()  == torch::kLong,
                    "TensorDual::index_select: index tensor must be int64");
        TORCH_CHECK(index.device() == r.device(),
                    "TensorDual::index_select: index tensor is on ", index.device(),
                    " but data are on ", r.device());
    
        /* gather real and dual rows ------------------------------------- */
        auto r_out = r.index_select(/*dim=*/0, index);          // (K)
        auto d_out = d.index_select(/*dim=*/0, index);          // (K,D)
    
        /* re-insert a length-1 state axis so constructor accepts shapes */
        r_out = r_out;                 // (K) is fine – constructor accepts (N)
        d_out = d_out;                 // (K,D)
    
        return TensorDual(std::move(r_out), std::move(d_out));  // invariant re-checked
    }
    
    /// Extract a single state element by row index.
    /// Shapes of the result stay valid for the class invariant:
    ///   r_out : (1)
    ///   d_out : (1, D)
    [[nodiscard]] inline TensorDual index_row(int64_t i) const
    {
        TORCH_CHECK(i >= 0 && i < r.size(0),
                    "TensorDual::index_row: index ", i, " out of range [0,", r.size(0)-1, ']');

        /* pick the same row from r and d ----------------------------------- */
        auto r_out = r.index_select(/*dim=*/0, torch::tensor({i}, r.options().dtype(torch::kLong)));
        auto d_out = d.index_select(/*dim=*/0, torch::tensor({i}, d.options().dtype(torch::kLong)));

        return TensorDual(std::move(r_out), std::move(d_out));   // shapes (1) / (1,D)
    }



    /// In-place masked assignment (row mask on state axis).
    /// `mask` shape : (N) bool
    /// `value` must share dtype/device and dual length D.
    inline void index_put_(const torch::Tensor& mask,
        const TensorDual&   value)
    {
        TORCH_CHECK(mask.dtype()  == torch::kBool,
        "index_put_: mask must be a boolean tensor");
        TORCH_CHECK(mask.device() == r.device(),
        "index_put_: device mismatch (mask on ", mask.device(),
        " vs data on ", r.device(), ')');
        TORCH_CHECK(mask.dim() == 1 && mask.size(0) == r.size(0),
        "index_put_: mask must be 1-D and length N=", r.size(0));

        TORCH_CHECK(value.D_ == d.size(1),
        "index_put_: dual-axis length mismatch (", value.D_,
        " vs ", d.size(1), ')');

        /* expand mask for (N,D) tensor ---------------------------------- */
        auto mask_d = mask.unsqueeze(-1).expand({-1, d.size(1)});

        /* in-place updates ---------------------------------------------- */
        r.index_put_({mask}, value.r.index({mask}));
        d.index_put_({mask_d}, value.d.index({mask_d}));
    }

        /// In-place assignment with a single `TensorIndex` applied **on the state
        /// axis 0**.  
        /// `value` must share dtype/device and have the same dual-axis length D.
        ///
        /// Supported selectors (`mask`):
        ///   • integer          → write one row  
        ///   • slice / range    → write several rows  
        ///   • boolean 1-D mask → write where mask == true
        inline void index_put_(const torch::indexing::TensorIndex& mask,
            const TensorDual&                   value)
        {
            using torch::indexing::Ellipsis;
            using torch::indexing::Slice;

            torch::Tensor row_mask; // bool mask (N)

            /* 1. scalar index -------------------------------------------------- */
            if (mask.is_integer()) {
                /* SymInt  →  concrete int64_t                          */
                int64_t i = mask.integer().expect_int();   // <-- here
            
                if (i < 0) i += r.size(0);
                TORCH_CHECK(i >= 0 && i < r.size(0),
                            "index_put_: index ", i, " out of range");
            
                row_mask = torch::zeros(r.sizes(),
                                        torch::TensorOptions()
                                            .dtype(torch::kBool)
                                            .device(r.device()));
                row_mask[i] = true;
            }
            /* 2. slice / ellipsis --------------------------------------------- */
            else if (mask.is_slice() || mask.is_ellipsis())
            {
                row_mask = torch::zeros(r.sizes(),
                                        torch::TensorOptions()
                                            .dtype(torch::kBool)
                                            .device(r.device()));
                row_mask.index_put_({mask}, true); // TensorIndex works here
            }
            /* 3. boolean tensor mask ------------------------------------------ */
            else if (mask.is_tensor()) {                     // ← boolean-mask case
                row_mask = mask.tensor();                    // returns const Tensor&
                TORCH_CHECK(row_mask.dtype()  == torch::kBool &&
                            row_mask.device() == r.device()   &&
                            row_mask.sizes()  == r.sizes(),
                            "index_put_: boolean mask must be kBool, same device, and shape (N)");
            }            
            else
            {
                TORCH_CHECK(false, "index_put_: unsupported TensorIndex type");
            }

            /* write selected rows --------------------------------------------- */
            r.index_put_({row_mask}, value.r.index({row_mask}));

            auto row_mask_d = row_mask.unsqueeze(-1).expand({-1, d.size(1)}); // (N,D)
            d.index_put_({row_mask_d}, value.d.index({row_mask_d}));
        }

    /// Masked in-place assignment with a **boolean row mask** and a scalar.
    ///
    /// `mask` shape : (N) bool   – selects rows on the state axis  
    /// `value`      : arithmetic scalar compatible with `r`’s dtype
    template <typename Scalar,
            typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
    inline void index_put_(const torch::Tensor& mask, Scalar value)
    {
        /* ── mask checks ─────────────────────────────────────────────── */
        TORCH_CHECK(mask.dtype()  == torch::kBool,
                    "index_put_: mask must be a boolean tensor");
        TORCH_CHECK(mask.device() == r.device(),
                    "index_put_: mask is on ", mask.device(),
                    " but tensors are on ", r.device());
        TORCH_CHECK(mask.sizes()  == r.sizes(),
                    "index_put_: mask shape ", mask.sizes(),
                    " must match real tensor shape ", r.sizes());

        /* ── update real part (broadcasts scalar) ────────────────────── */
        r.index_put_({mask}, value);

        /* ── update dual part: set rows to zeros ─────────────────────── */
        auto mask_d = mask.unsqueeze(-1).expand({-1, d.size(1)}); // (N,D)
        d.index_put_({mask_d}, 0.0);                              // preserves dtype
    }

    /// In-place masked assignment with a scalar `value`.
    ///
    /// `mask`  ─ bool tensor of shape **(N)** on the same device  
    /// real part  r[mask] ← value  
    /// dual part  d[mask, :] ← 0
    inline void index_put_(const torch::Tensor& mask, double value)
    {
        // ── basic checks ───────────────────────────────────────────────
        TORCH_CHECK(mask.dtype()  == torch::kBool,
                    "index_put_: mask must be a boolean tensor");
        TORCH_CHECK(mask.device() == r.device(),
                    "index_put_: mask on ", mask.device(),
                    " but data on ", r.device());
        TORCH_CHECK(mask.dim() == 1 && mask.size(0) == r.size(0),
                    "index_put_: mask shape ", mask.sizes(),
                    " must be (N) with N = ", r.size(0));

        // ── cast scalar to tensor’s dtype (keeps fp32/fp64/complex) ────
        auto scalar_val = torch::scalar_tensor(
                            value,
                            r.options().dtype(r.dtype()))             // same dtype / device
                            .item();                                   // convert back to C++ scalar

        // ── update real part ------------------------------------------
        r.index_put_({mask}, scalar_val);

        // ── update dual part  (zero the matching rows) -----------------
        auto mask_d = mask.unsqueeze(-1)             // (N,1)
                        .expand({-1, d.size(1)}); // (N,D)
        d.index_put_({mask_d}, 0);                   // literal 0 → correct dtype automatically
    }

    /// In-place assignment with an *advanced-index vector* applied on the
    /// **state axis 0 only**.  
    /// Every `TensorIndex` in `sel` is interpreted exactly as you would pass
    /// to `x.index(sel)` in PyTorch.
    ///
    /// - `value` must share dtype/device with the target tensors.
    /// - `value.D_` must equal the current dual-axis length **D**.
    /// - The slice produced by `r.index(sel)` must have the **same shape** as
    ///   `value.r`, and likewise for the dual slice vs `value.d`.
    inline void
    index_put_(const std::vector<torch::indexing::TensorIndex>& sel,
            const TensorDual&                                value)
    {
        using torch::indexing::TensorIndex;

        /* ── basic parity checks ───────────────────────────────────────── */

        TORCH_CHECK(value.D_ == d.size(1),
                    "index_put_: dual-axis length mismatch (",
                    value.D_, " vs ", d.size(1), ')');

        /* ── materialise the selector on `r` once to know its shape ────── */
        auto r_slice = r.index(sel);            // view — no copy
        TORCH_CHECK(r_slice.sizes() == value.r.sizes(),
                    "index_put_: value.r shape ", value.r.sizes(),
                    " does not match selected region ", r_slice.sizes());

        /* ── same check for the dual tensor ────────────────────────────── */
        auto d_slice = d.index(sel);            // view
        TORCH_CHECK(d_slice.sizes() == value.d.sizes(),
                    "index_put_: value.d shape ", value.d.sizes(),
                    " does not match selected region ", d_slice.sizes());

        /* ── in-place writes (PyTorch handles broadcasting rules) ──────── */
        r.index_put_(sel, value.r);
        d.index_put_(sel, value.d);
    }

    /// In-place masked write of a scalar.
    ///
    /// `sel`  – vector of TensorIndex objects **for the state axis 0 only**  
    /// `value`– scalar convertible to `r.dtype()`  
    /// real part:  r[sel] ← value  
    /// dual part:  d[sel, :] ← 0
    inline void index_put_(const std::vector<torch::indexing::TensorIndex>& sel,
        double                                          value)
    {
        /* sanity: selector applies only to axis-0 */
        TORCH_CHECK(sel.size() == 1,
        "index_put_(scalar): selector must address the state axis only");

        /* cast scalar to the tensor’s dtype (keeps fp32/fp64/complex) */
        auto scalar_val = torch::scalar_tensor(
            value,
            r.options().dtype(r.dtype()) )      // correct dtype / device
            .item();                            // back to C++ scalar

        /* write real part ------------------------------------------------ */
        r.index_put_(sel, scalar_val);

        /* build a selector for the dual tensor: same rows, all D columns */
        using namespace torch::indexing;
        auto sel_d = sel;                 // copy row selector
        sel_d.push_back(Slice());         // add ':' for the dual axis

        /* zero the dual slots ------------------------------------------- */
        d.index_put_(sel_d, 0);           // literal 0 promoted to d.dtype()
    }

    /// Element-wise maximum of two TensorDuals.
    ///
    ///   r_out = max(r₁ , r₂)
    ///   d_out = (r₁ > r₂)  ?  d₁ : d₂          // tie → take d₂
    ///
    /// Requirements  
    ///   • same dtype / device  
    ///   • same state length **N** and dual length **D**
    [[nodiscard]] inline TensorDual max(const TensorDual& other) const noexcept
    {
        /* ── compatibility checks ─────────────────────────────────────── */
        TORCH_CHECK(r.sizes()   == other.r.sizes(),
                    "TensorDual::max: state-axis length mismatch");
        TORCH_CHECK(d.sizes()   == other.d.sizes(),
                    "TensorDual::max: dual-tensor shape mismatch");

        /* ── element-wise maximum on the real part ────────────────────── */
        auto r_out = torch::maximum(r, other.r);          // (N)

        /* ── choose dual slot from the same index ----------------------- */
        auto choose_first = (r > other.r).unsqueeze(-1);  // (N,1)
        auto d_out        = torch::where(choose_first, d, other.d);  // (N,D)

        return TensorDual(std::move(r_out), std::move(d_out));       // invariant re-checked
    }


    /// Raise a TensorDual to a **scalar** power `p`.
    ///
    ///   y  = r^p  
    ///   dy = p · r^(p-1) · d              (element-wise, broadcast over D)
    ///
    /// * If `r` contains negatives and `p` is not an integer, the result is
    ///   promoted to the matching complex dtype so that `x^p` is well-defined.
    /// * Works for any real/complex float dtype; integer tensors are rejected.
    [[nodiscard]] inline TensorDual pow(double p) const
    {
        TORCH_CHECK(r.is_floating_point() || r.is_complex(),
                    "TensorDual::pow: only floating/complex dtypes supported");

        const bool frac_exp = (p != std::nearbyint(p));

        /* ── promote to complex if needed (negative base & fractional p) ───── */
        auto r_base = r;
        auto d_base = d;
        if (frac_exp && (!r.is_complex()) && (r < 0).any().item<bool>())
        {
            auto cdtype = torch::toComplexType(r.scalar_type());
            r_base = r.to(cdtype);
            d_base = d.to(cdtype);
        }

        /* ── primal  r^p ───────────────────────────────────────────────────── */
        auto r_pow = torch::pow(r_base, p);               // (N)

        /* ── derivative  p·r^(p-1)·d  (broadcast over dual axis D) ─────────── */
        auto grad  = p * torch::pow(r_base, p - 1.0);     // (N)
        auto d_pow = grad.unsqueeze(-1) * d_base;         // (N,D)

        return TensorDual(std::move(r_pow), std::move(d_pow));   // invariant re-checked
    }



    TensorMatDual unsqueeze(int dim); 
    TensorMatDual eye();
};















}


