
import sys
import shutil
from pathlib import Path

project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))

import pybind11_stubgen


cpp_module_name = '_C'

    
def main():
    stubs_path = project_path / f'synapx/{cpp_module_name}'
    
    # Remove previous stubs
    if stubs_path.exists():
        shutil.rmtree(stubs_path)
    
    # Generate new stubs
    sys.argv = [None, f'synapx.{cpp_module_name}', '--output', '.']
    pybind11_stubgen.main()
    
    # Adjust generated stubs
    if stubs_path.is_dir():
        for stub_file in stubs_path.iterdir():
            
            if stub_file.name != '__init__.pyi': 
                # Adjustments only needed for the main module stub file at the moment
                continue
            
            print(f"[%] Adjusting content for '{stub_file.name}'")
            
            with open(stub_file, 'r') as file:
                content = file.read()
                
                # Add torch import
                content = content.replace("import numpy", "import torch\nimport numpy")
                
                # Replace wrong return value 
                content = content.replace(
                    "def grad(self) -> typing.Any:", 
                    "def grad(self) -> Tensor | None:"
                )
                
                content = content.replace(
                    "def grad_fn(self) -> typing.Any:", 
                    "def grad_fn(self) -> Node | None:"
                )
                
                # Wrong generated type
                content = content.replace("numpy.dtype[typing.Any]", "torch.dtype")
                
                # Improve device annotation
                content = content.replace(": device", ": torch.device")
                
            with open(stub_file, 'w') as file:
                file.write(content)
    else:
        print(f"[!] Stubs directory '{cpp_module_name}' does not exist")


if __name__ == "__main__":
    main()