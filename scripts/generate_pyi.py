
import sys
import subprocess
from pathlib import Path


project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))
    
    
def main():
    command = ['pybind11-stubgen', 'synapx._C', '--output', '.']
    subprocess.run(command, cwd=project_path)
    
    stub_path = project_path / 'synapx/_C.pyi'
    if stub_path.exists():
        with open(stub_path, 'r') as file:
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
            
        with open(stub_path, 'w') as file:
            file.write(content)
    else:
        print("[!] Stub file does not exist")


if __name__ == "__main__":
    main()