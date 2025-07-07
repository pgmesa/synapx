import os
from pathlib import Path
from setuptools import setup, Extension


cwd = Path(__file__).parent.absolute()

stub_c = """
#include <Python.h>

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  return NULL;
}
"""

# Minimal stub
stub_file = cwd / 'stub.c'
with open(stub_file, 'w') as f:
    f.write(stub_c)

synapx_lib_dir = cwd / 'synapx/lib'
lib_dirs = [str((synapx_lib_dir/d).relative_to(cwd)) for d in os.listdir(synapx_lib_dir)]

extensions = [
    Extension(
        'synapx._C',
        libraries=['synapx'],
        language="C++",
        sources=['stub.c'],
        library_dirs=lib_dirs
    )
]

setup(
    ext_modules=extensions,
)