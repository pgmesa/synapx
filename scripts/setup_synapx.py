
import sys
import shutil
import platform
from pathlib import Path

import torch
import pybind11_stubgen


project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))

package_name = "synapx"
interface_name = '_C'

build_path = project_path / 'libsynapx/build'


def main():
    nchars = 50
    print("-"*nchars)
    print("Setting up SynapX Python Package".center(nchars))
    print("-"*nchars)
    
    system_name = platform.system()
    print(f"Detected '{system_name}' system")

    if system_name == "Windows":
        dll_path = build_path / f'Release/{package_name}.dll'
        import_lib_path = build_path / f'Release/{package_name}.lib'

        bindings_path = build_path / 'bindings/Release'
        interface_ext = '.pyd'
    elif system_name == 'Linux':
        dll_path = build_path / f"lib{package_name}.so"
        import_lib_path = build_path / f"lib{package_name}.a"

        bindings_path = build_path / 'bindings'
        interface_ext = '.so'
    else:
        print(f"Unsupported system: {system_name}")

    for f in bindings_path.glob(f'{interface_name}.*{interface_ext}'):
        bindings_path /= f
        break
    else:
        print(f"[x] Error: No *.{interface_ext} file detected")

    package_path = project_path / package_name

    torch_version = '.'.join(torch.__version__.split('.')[:2])
    lib_path = package_path / f'lib/libtorch-{torch_version}.x'
    stub_dest_path = package_path / f'{package_name}/{interface_name}.pyi'
    
    print("Library destination path:", lib_path)
    lib_path.mkdir(exist_ok=True, parents=True)
    
    # Copy Python-C++ interface
    shutil.copy(bindings_path, package_path)

    # Copy synapx shared libraries
    shutil.copy(dll_path, lib_path)
    if import_lib_path.exists():
        shutil.copy(import_lib_path, lib_path)

    # Generate stubs with pybind11-stubgen
    sys.argv = ['None', 'synapx._C', '--output', str(package_path)]
    pybind11_stubgen.main()

    shutil.copy(stub_dest_path, stub_dest_path.parent.parent)
    shutil.rmtree(stub_dest_path.parent)
    

if __name__ == "__main__":
    main()