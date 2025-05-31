
import sys
import subprocess
from pathlib import Path


project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))
    
    
def main():
    command = ['pybind11-stubgen', 'synapx._C', '--output', '.']
    subprocess.run(command, cwd=project_path)


if __name__ == "__main__":
    main()