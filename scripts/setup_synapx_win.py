
import sys
import shutil
from pathlib import Path
import pybind11_stubgen

# ----------------- Valid for Windows libsynapx build -----------------
# ---------------------------------------------------------------------

project_path = Path(__file__).parent.parent.absolute()
package_name = "synapx"
interface_name = '_C'

build_path = project_path / 'libsynapx/build'
dll_path = build_path / f'Release/{package_name}.dll'
import_lib_path = build_path / f'Release/{package_name}.lib'

bindings_path = build_path / 'bindings/Release'
for f in bindings_path.glob(f'{interface_name}.*.pyd'):
    bindings_path /= f
    break

package_path = project_path / package_name
lib_path = package_path / 'lib'
stub_dest_path = package_path / f'{package_name}/{interface_name}.pyi'


def main():
    lib_path.mkdir(exist_ok=True)
    
    # Copy Python-C++ interface
    shutil.copy(bindings_path, package_path)

    # Copy synapx shared libraries
    shutil.copy(dll_path, lib_path)
    shutil.copy(import_lib_path, lib_path)

    # Generate stubs with pybind11-stubgen
    sys.argv += ['synapx._C', '--output', str(package_path)]
    pybind11_stubgen.main()

    shutil.copy(stub_dest_path, stub_dest_path.parent.parent)
    shutil.rmtree(stub_dest_path.parent)
    

if __name__ == "__main__":
    main()