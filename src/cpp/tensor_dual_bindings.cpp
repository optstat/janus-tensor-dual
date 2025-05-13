// tensor_dual_bindings.cpp
#include <torch/extension.h>
#include "tensordual.hpp"
#include <pybind11/pybind11.h>
#include "bindings_fwd.hpp" // forward declarations for the bindings
using janus::TensorDual;
using TDPtr = c10::intrusive_ptr<janus::TensorDual>;
TORCH_LIBRARY(janus, m)
{
    m.class_<TensorDual>("TensorDual")
        // tensor_dual_bindings.cpp  – inside the class_ block
        .def("r", [](const TDPtr& self) { return self->r; })
        .def("d", [](const TDPtr& self) { return self->d; })
        // — ctor —
        .def(torch::init<const torch::Tensor &, const torch::Tensor &>())

        // — attribute-style getters implemented as 0-arg methods —
        .def("__getattr__",
             [](const TDPtr &self, const std::string &name)
             {
                 if (name == "r")
                     return self->r;
                 if (name == "d")
                     return self->d;
                 throw std::runtime_error("No such attribute: " + name);
             })

        // — repr —
        .def("__repr__",
             [](const TDPtr &self)
             {
               std::ostringstream ss; ss << *self; return ss.str(); })

        // — binary ops (always return new intrusive_ptr) —
        .def("__add__",
             [](const TDPtr &a, const TDPtr &b)
             {
                 return c10::make_intrusive<TensorDual>(a->operator+(*b));
             })
        .def("__radd__",
             [](const TDPtr &b, const TDPtr &a)
             {
                 return c10::make_intrusive<TensorDual>(a->operator+(*b));
             })

        .def("__sub__", [](const TDPtr &a, const TDPtr &b)
             { return c10::make_intrusive<TensorDual>(a->operator-(*b)); })
        .def("__mul__", [](const TDPtr &a, const TDPtr &b)
             { return c10::make_intrusive<TensorDual>(a->operator*(*b)); })
        .def("__truediv__", [](const TDPtr &a, const TDPtr &b)
             { return c10::make_intrusive<TensorDual>(a->operator/(*b)); })

        // — a couple of methods —
        .def("square", [](const TDPtr &self)
             { return c10::make_intrusive<TensorDual>(self->square()); })
        .def_static("zeros",
                    [](int64_t N, int64_t D,
                       torch::Device dev, torch::Dtype dt)
                    {
                        return c10::make_intrusive<TensorDual>(
                            TensorDual::zeros(N, D, dev, dt));
                    }) // ← no py::arg

        .def_static("eye",
                    [](int64_t N,
                       torch::Dtype dt, torch::Device dev)
                    {
                        return c10::make_intrusive<TensorDual>(
                            TensorDual::eye(N, dt, dev));
                    });
}

namespace janus_bind
{
    void add_tensor_dual_bindings(pybind11::module_& m)
    {
    }
    void add_tensor_dual_alias(pybind11::module_& m)
    {
        namespace py = pybind11;
        py::object torch_classes =
            py::module_::import("torch").attr("classes");
        m.attr("TensorDual") =
            torch_classes.attr("janus").attr("TensorDual");
    }
} // namespace janus_bind
