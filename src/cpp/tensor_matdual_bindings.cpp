/*───────────────────────────────────────────────────────────*/
/*  tensor_matdual_bindings.cpp                              */
/*───────────────────────────────────────────────────────────*/
#include "tensormatdual.hpp" // <- your header
#include "bindings_fwd.hpp"
#include <pybind11/pybind11.h>
#include <torch/script.h> // torch::class_
namespace py = pybind11;
namespace janus
{
    using TMD = TensorMatDual;
    using TMDPtr = c10::intrusive_ptr<TMD>;
} // namespace janus
TORCH_LIBRARY_FRAGMENT(janus, m)
{
    using TMD = janus::TensorMatDual;
    using TMDPtr = c10::intrusive_ptr<TMD>;

    // ---- 1. register the custom class ---------------------------
    auto cls = m.class_<TMD>("TensorMatDual") // <-- this is the line that matters
                   .def(torch::init<const torch::Tensor &,
                                    const torch::Tensor &>())

                   .def_readonly("r", &TMD::r)
                   .def_readonly("d", &TMD::d)
                   .def_static(
                       "createZero",
                       [](const torch::Tensor &r, int64_t D)
                       {
                           return c10::make_intrusive<TMD>(TMD::createZero(r, D));
                       })

                   .def("square",
                        [](const TMDPtr &self)
                        {
                            return c10::make_intrusive<TMD>(self->square());
                        })

                   .def("__repr__",
                        [](const TMDPtr &self)
                        {
                            std::ostringstream ss;
                            ss << *self;
                            return ss.str();
                        })
                   /* --------------- basic ops ------------------------ */
                   .def("__add__", [](const TMDPtr &a, const TMDPtr &b)
                        { return c10::make_intrusive<TMD>(*a + *b); })
                   .def("__radd__", [](const TMDPtr &b, const TMDPtr &a)
                        { return c10::make_intrusive<TMD>(*a + *b); })

                   .def("__sub__", [](const TMDPtr &a, const TMDPtr &b)
                        { return c10::make_intrusive<TMD>(*a - *b); })
                   .def("__neg__", [](const TMDPtr &a)
                        { return c10::make_intrusive<TMD>(-*a); })

                   .def("__mul__", [](const TMDPtr &a, double c)
                        { return c10::make_intrusive<TMD>(*a * c); })

                   .def("__truediv__", [](const TMDPtr &a, double c)
                        { return c10::make_intrusive<TMD>(*a / c); })
                   .def("eye",
                        [](const TMDPtr &self)
                        {
                            // Call the C++ member and wrap its result
                            return c10::make_intrusive<TMD>(self->eye());
                        })
                   /* --------------- helpers -------------------------- */
                   .def("rows", &TMD::rows)
                   .def("cols", &TMD::cols)
                   .def("dual_dim", &TMD::dual_dim)
                   .def("clone", [](const TMDPtr &a)
                        { return c10::make_intrusive<TMD>(a->clone()); })
                   .def("to", [](TMDPtr self, torch::Device dev) { //  ← fixed
                       self->to_(dev);
                       return self; //  optional; keep if you want chaining
                   });
}
namespace janus_bind
{
    using janus::TensorMatDual;
    using TMDPtr = c10::intrusive_ptr<TensorMatDual>;
    
    /* 2.a  Pybind wrapper: keyword args + nice __repr__ + operator tags */
    void add_tensor_matdual_bindings(py::module_& m)
    {
        auto cls = py::class_<TensorMatDual, TMDPtr>(m, "TensorMatDual")
           .def(py::init<const at::Tensor&, const at::Tensor&>(),
                py::arg("real"), py::arg("dual"),
                "real: [N,L]  • dual: [N,L,D]")
    
           .def("__repr__", [](const TMDPtr& self){
                  std::ostringstream ss; ss << *self; return ss.str(); })
    
           .def("__add__", [](const TMDPtr& a, const TMDPtr& b){
                  return c10::make_intrusive<TensorMatDual>(*a + *b);
                }, py::is_operator())
    
           .def("eye", [](const TMDPtr& self){
                  return c10::make_intrusive<TensorMatDual>(self->eye());
                });
           /* add more Python-only sugar if you like */
    }
    
    void add_tensor_matdual_alias(pybind11::module_ &m)
    {
        //namespace py = pybind11;
        //py::object torch_classes =
        //    py::module_::import("torch").attr("classes");
        //m.attr("TensorMatDual") =
        //    torch_classes.attr("janus").attr("TensorMatDual");
    }
}
