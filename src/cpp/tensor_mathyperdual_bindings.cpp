// ──────────────────────────────────────────────────────────────
//  tensor_mathyperdual_bindings.cpp
// ──────────────────────────────────────────────────────────────
#include <torch/extension.h>
#include "tensormathyperdual.hpp"      //  <--  your header
#include <sstream>

namespace janus {
using TMH   = TensorMatHyperDual;
using TMHPtr= c10::intrusive_ptr<TMH>;
} // namespace janus

TORCH_LIBRARY(janus, m)
{
    using namespace janus;
    namespace py = pybind11;

    /*──────── 1.  Dispatcher-side custom-class registration ───────*/
    auto cls = torch::class_<TMH>(m, "TensorMatHyperDual")
        // python ctor:  janus.TensorMatHyperDual(r, d, h)
        .def(torch::init<const torch::Tensor&,
                         const torch::Tensor&,
                         const torch::Tensor&>())

        // plain tensor accessors (read-only views)
        .def_readonly("r", &TMH::r)
        .def_readonly("d", &TMH::d)
        .def_readonly("h", &TMH::h)

        // factory:   janus.TensorMatHyperDual.createZero(r, D)
        .def_static("createZero",
            [](const torch::Tensor& r, int64_t D)
            {
                return c10::make_intrusive<TMH>(TMH::createZero(r, D));
            },
            py::arg("real"), py::arg("dual_dim"))

        // quick unary op – returns new intrusive_ptr
        .def("square",
            [](const TMHPtr& self)
            {
                return c10::make_intrusive<TMH>(self->square());
            })

        // repr
        .def("__repr__",
            [](const TMHPtr& self)
            {
                std::ostringstream ss; ss << *self; return ss.str();
            })

        //  x + y
        .def("__add__",
            [](const TMHPtr& a, const TMHPtr& b)
            {
                return c10::make_intrusive<TMH>(*a + *b);
            },
            py::is_operator())

        //  x - y
        .def("__sub__",
            [](const TMHPtr& a, const TMHPtr& b)
            {
                return c10::make_intrusive<TMH>(*a - *b);
            },
            py::is_operator());
}

