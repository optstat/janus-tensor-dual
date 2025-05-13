#pragma once
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace janus_bind {
    void add_tensor_dual_bindings(pybind11::module_&);    
    void add_tensor_dual_alias(pybind11::module_&);   
    void add_tensor_hyperdual_bindings(pybind11::module_&);
    void add_tensor_hyperdual_alias(pybind11::module_&);       
    void add_tensor_matdual_bindings(py::module_&);   
    void add_tensor_matdual_alias(py::module_&);
    void add_tensor_mathyperdual_bindings(py::module_&);
    void add_tensor_mathyperdual_alias(py::module_&); // registers the class 
}