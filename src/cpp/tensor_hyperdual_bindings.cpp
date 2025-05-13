// tensor_hyperdual_bindings.cpp
#include <torch/extension.h>
#include "tensordual.hpp"         // needs TensorDual for the promotion-ctor
#include "tensorhyperdual.hpp"    // the header you pasted
#include "bindings_fwd.hpp"    // forward declarations for the bindings
namespace py   = pybind11;
using   TDHPtr = c10::intrusive_ptr<janus::TensorHyperDual>;
using   TDPtr  = c10::intrusive_ptr<janus::TensorDual>;

/*──────────────────────── 1.  Torch custom-class registry ────────────────*/
TORCH_LIBRARY_FRAGMENT(janus, m)
{
    // Register as torch.classes.janus.TensorHyperDual
    m.class_<janus::TensorHyperDual>("TensorHyperDual")

        /* ——— ctor: (r,d,h) ——— */
        .def(torch::init<const torch::Tensor&,
                         const torch::Tensor&,
                         const torch::Tensor&>())

        /* ——— zero-args inspectors ——— */
        .def("r",  [](const TDHPtr& self){ return self->r; })
        .def("d",  [](const TDHPtr& self){ return self->d; })
        .def("h",  [](const TDHPtr& self){ return self->h; })

        /* ——— static helpers (positional args only) ——— */
        .def_static("zeros",
            [](int64_t N,int64_t D,
               torch::Dtype dt, torch::Device dev){
                return c10::make_intrusive<janus::TensorHyperDual>(
                         janus::TensorHyperDual::zeros(N,D,dt,dev));
            })

        .def_static("eye",
            [](int64_t N,
               torch::Dtype dt, torch::Device dev){
                // Build eye via TensorDual promotion, then construct TDH
                auto td = janus::TensorDual::eye(N,dt,dev);
                return c10::make_intrusive<janus::TensorHyperDual>(td);
            })

        /* ——— unary ops you’ll actually call soon ——— */
        .def("__neg__",   [](const TDHPtr& a){
                return c10::make_intrusive<janus::TensorHyperDual>(-(*a));
            })

        .def("square",    [](const TDHPtr& a){
                return c10::make_intrusive<janus::TensorHyperDual>(a->square());
            })

        .def("sqrt",      [](const TDHPtr& a){
                return c10::make_intrusive<janus::TensorHyperDual>(a->sqrt());
            })

        /* ——— binary +,-,*,/ ——— */
        .def("__add__",   [](const TDHPtr& a,const TDHPtr& b){
                return c10::make_intrusive<janus::TensorHyperDual>(*a + *b);
            })
        .def("__sub__",   [](const TDHPtr& a,const TDHPtr& b){
                return c10::make_intrusive<janus::TensorHyperDual>(*a - *b);
            })
        .def("__mul__",   [](const TDHPtr& a,const TDHPtr& b){
                return c10::make_intrusive<janus::TensorHyperDual>(*a * *b);
            })
        .def("__truediv__",[](const TDHPtr& a,const TDHPtr& b){
                return c10::make_intrusive<janus::TensorHyperDual>(*a / *b);
            })

        /* ——— string repr ——— */
        .def("__repr__",  [](const TDHPtr& self){
                std::ostringstream ss; ss << *self; return ss.str();
            });
}
namespace janus_bind
{
    // Register the TensorHyperDual class in the janus namespace
    void add_tensor_hyperdual_bindings(pybind11::module_& m){
        //For future python bindings
    }
void add_tensor_hyperdual_alias(pybind11::module_& m)
{
    namespace py = pybind11;
    py::object torch_classes =
        py::module_::import("torch").attr("classes");
    m.attr("TensorHyperDual") =
        torch_classes.attr("janus").attr("TensorHyperDual");
}
}