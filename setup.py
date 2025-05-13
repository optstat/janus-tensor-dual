from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
cpp_dir  = Path("src/cpp")
all_cpp = sorted([str(p) for p in cpp_dir.glob("*.cpp")])
all_cpp.remove("src/cpp/janus_module.cpp")          # pop it out
sources   = ["src/cpp/janus_module.cpp"] + all_cpp  # put it back in front  
setup(
    name="janus_dual",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="janus",                              # import janus
            sources=sources,
            include_dirs=[str(cpp_dir)],               # so #include "TensorDual.h" works
            extra_compile_args = {
                "cxx": ["-std=c++17", "-O3", "-Wno-sign-compare"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)