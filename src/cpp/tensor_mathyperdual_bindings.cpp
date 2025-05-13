// ──────────────────────────────────────────────────────────────
//  tensor_mathyperdual_bindings.cpp
// ──────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include "tensormathyperdual.hpp"      //  <--  your header
#include <sstream>
#include "tensormathyperdual.hpp"
#include "bindings_fwd.hpp" // forward declarations for the bindings
namespace janus {
    using TMH   = TensorMatHyperDual;
    using TMHPtr= c10::intrusive_ptr<TMH>;
} // namespace janus

TORCH_LIBRARY_FRAGMENT(janus, m) {
    using TMH    = janus::TensorMatHyperDual;
    using TMHPtr = c10::intrusive_ptr<TMH>;

    // ---- 1. register the custom class ---------------------------
    auto cls = m.class_<TMH>("TensorMatHyperDual")   // <-- this is the line that matters
        .def(torch::init<const torch::Tensor&,
                         const torch::Tensor&,
                         const torch::Tensor&>())

        .def_readonly("r", &TMH::r)
        .def_readonly("d", &TMH::d)
        .def_readonly("h", &TMH::h)
        .def_static(
            "createZero",
            [](const torch::Tensor& r, int64_t D) {
                return c10::make_intrusive<TMH>(TMH::createZero(r, D));
            })

        .def("square",
             [](const TMHPtr& self) {
                 return c10::make_intrusive<TMH>(self->square());
             })

        .def("__repr__",
             [](const TMHPtr& self) {
                 std::ostringstream ss; ss << *self; return ss.str();
             })

        .def("__add__", [](const TMHPtr& a, const TMHPtr& b) {
                 return c10::make_intrusive<TMH>(*a + *b);
             })

        .def("__sub__", [](const TMHPtr& a, const TMHPtr& b) {
                 return c10::make_intrusive<TMH>(*a - *b);
             });
}
namespace janus_bind
{
    void add_tensor_mathyperdual_bindings(pybind11::module_& m)
    {
    }
void add_tensor_mathyperdual_alias(pybind11::module_& m)
{
    namespace py = pybind11;
    py::object torch_classes =
        py::module_::import("torch").attr("classes");
    m.attr("TensorMatHyperDual") =
        torch_classes.attr("janus").attr("TensorMatHyperDual");
}
}