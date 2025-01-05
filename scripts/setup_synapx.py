
import sys
import shutil
import platform
import subprocess
from pathlib import Path


project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))
    
package_name = "synapx"
interface_name = '_C'

build_path = project_path / 'libsynapx/build'

package_path = project_path / package_name
package_lib_path = package_path / 'lib'


def get_torch_version(python_exe):
    if python_exe is not None:
        # Command to get the PyTorch version
        command = [python_exe, "-c", "import torch; print(torch.__version__)"]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            torch_version = result.stdout.strip()
            print(f"PyTorch version in the target environment: {torch_version}")
        except subprocess.CalledProcessError as e:
            print(f"Error while running the command: {e}")
        except FileNotFoundError:
            print("The specified Python interpreter was not found.")
    else:
        import torch
        torch_version = torch.__version__ 

    return torch_version    

def main(python_exe=None, gen_stubs=True):
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

    torch_version = get_torch_version(python_exe)
    torch_version = '.'.join(torch_version.split('.')[:2])
    lib_path = package_lib_path / f'libtorch-{torch_version}.x'
    stub_dest_path = package_path / f'{package_name}/{interface_name}.pyi'
    
    print("Library destination path:", lib_path)
    lib_path.mkdir(exist_ok=True, parents=True)
    
    # Copy Python-C++ interface
    shutil.copy(bindings_path, lib_path)

    # Copy synapx shared libraries
    shutil.copy(dll_path, lib_path)
    if import_lib_path.exists():
        shutil.copy(import_lib_path, lib_path)

    # Generate stubs with pybind11-stubgen
    if gen_stubs:
        stub_args = ['None', 'synapx._C', '--output', str(package_path)]
        if python_exe is not None:
            gen_command = f"sys.argv = {stub_args}; pybind11_stubgen.main()"
            command = [python_exe, '-c', f"import pybind11_stubgen, sys; {gen_command}"]
            subprocess.run(command, check=True)
        else:
            import pybind11_stubgen
            sys.argv = stub_args
            pybind11_stubgen.main()

        shutil.copy(stub_dest_path, stub_dest_path.parent.parent)
        shutil.rmtree(stub_dest_path.parent)
    

if __name__ == "__main__":
    main()