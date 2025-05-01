/*───────────────────────────────────────────────────────────*/
/*  tensor_matdual_bindings.cpp                              */
/*───────────────────────────────────────────────────────────*/
#include "tensormatdual.hpp"          // <- your header
#include <pybind11/pybind11.h>
#include <torch/script.h>             // torch::class_
namespace py = pybind11;
using janus::TensorMatDual;
using TMDPtr = c10::intrusive_ptr<TensorMatDual>;

void add_tensor_matdual_bindings(py::module_ &m)
{
    // ───────── register with Torch dispatcher ──────────────
    torch::class_<TensorMatDual>(m, "TensorMatDual")
        /* Python ctor:  janus.TensorMatDual(real, dual) */
        .def(torch::init<const torch::Tensor&,
                         const torch::Tensor&>(),
             py::arg("real"), py::arg("dual"))

        /* --------- attribute-style getters (0-arg) -------- */
        .def("__getattr__",
             [](const TMDPtr &self, const std::string &name)
             {
                 if (name == "r") return self->r;
                 if (name == "d") return self->d;
                 throw std::runtime_error("Unknown attribute '" + name + "'");
             })

        /* --------------- __repr__ ------------------------- */
        .def("__repr__",
             [](const TMDPtr &self)
             {
                 std::ostringstream ss; ss << *self; return ss.str();
             })

        /* --------------- factories ------------------------ */
        .def_static("create_zero",
            [](const torch::Tensor& r, int D)
            {
                return c10::make_intrusive<TensorMatDual>(
                        TensorMatDual::createZero(r, D));
            },
            py::arg("real"), py::arg("ddim"))

        .def_static("eye",
            [](int64_t N, int64_t L, int64_t D,
               torch::Dtype dt, torch::Device dev)
            {
                auto r = torch::eye(N, L,
                            torch::TensorOptions().dtype(dt).device(dev));
                auto d = torch::zeros({N, L, D},
                            torch::TensorOptions().dtype(dt).device(dev));
                return c10::make_intrusive<TensorMatDual>(r, d);
            },
            py::arg("N"), py::arg("L"), py::arg("D"),
            py::arg("dtype") = torch::kFloat64,
            py::arg("device")= torch::kCPU)

        /* --------------- basic ops ------------------------ */
        .def("__add__", [](const TMDPtr &a, const TMDPtr &b)
             { return c10::make_intrusive<TensorMatDual>(*a + *b); })
        .def("__radd__", [](const TMDPtr &b, const TMDPtr &a)
             { return c10::make_intrusive<TensorMatDual>(*a + *b); })

        .def("__sub__", [](const TMDPtr &a, const TMDPtr &b)
             { return c10::make_intrusive<TensorMatDual>(*a - *b); })
        .def("__neg__", [](const TMDPtr &a)
             { return c10::make_intrusive<TensorMatDual>(-*a); })

        .def("__mul__", [](const TMDPtr &a, double c)
             { return c10::make_intrusive<TensorMatDual>(*a * c); },
             py::is_operator())

        .def("__truediv__", [](const TMDPtr &a, double c)
             { return c10::make_intrusive<TensorMatDual>(*a / c); },
             py::is_operator())

        /* --------------- helpers -------------------------- */
        .def("rows",      &TensorMatDual::rows)
        .def("cols",      &TensorMatDual::cols)
        .def("dual_dim",  &TensorMatDual::dual_dim)
        .def("clone",     [](const TMDPtr &a)
             { return c10::make_intrusive<TensorMatDual>(a->clone()); })
        .def("to",        [](TMDPtr &a, torch::Device dev){ a->to_(dev); });

    /* ------ python-side convenience alias ---------------- */
    py::object torch_classes = py::module_::import("torch").attr("classes");
    m.attr("TensorMatDual") =
        torch_classes.attr("janus").attr("TensorMatDual");
}