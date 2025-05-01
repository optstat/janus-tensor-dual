from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

cpp_dir = Path("src/cpp")
cpp_src = [str(p) for p in cpp_dir.glob("*.cpp")]      # pick up every .cpp

setup(
    name="janus_dual",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="janus",                              # import janus
            sources=[
                 "src/cpp/tensor_dual_bindings.cpp",
                 "src/cpp/tensor_hyperdual_bindings.cpp",   
                  ],
            include_dirs=[str(cpp_dir)],               # so #include "TensorDual.h" works
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)