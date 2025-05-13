// janus_module.cpp
#include <torch/extension.h>
#include "bindings_fwd.hpp"

namespace py = pybind11;   // local shortcut – OK inside .cpp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace janus_bind;   // pull the fwd decls

    /* 1. Register the implementation classes ---------------------- */
    add_tensor_dual_bindings      (m);
    add_tensor_hyperdual_bindings (m);
    add_tensor_matdual_bindings   (m);
    add_tensor_mathyperdual_bindings(m);

    /* 2. Add python-side convenience aliases once ----------------- */
    add_tensor_dual_alias         (m);
    add_tensor_hyperdual_alias    (m);
    add_tensor_matdual_alias      (m);
    add_tensor_mathyperdual_alias (m);
}
