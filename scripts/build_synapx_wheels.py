
import os
import re
import sys
import time
import shutil
import platform
import subprocess
from pathlib import Path

import build_libsynapx
import setup_synapx


conda_envs_path = Path(os.environ['CONDA_ROOT']) / 'envs'


class Version:
    
    SEMVER_REGEX = re.compile(
        r"^(\d+)\.(\d+)\.(\d+)"      # Major, Minor, Patch
        r"(?:-([0-9A-Za-z.-]+))?"    # Pre-release
        r"(?:\+([0-9A-Za-z.-]+))?$"  # Build metadata
    )

    def __init__(self, version_str):
        match = self.SEMVER_REGEX.match(version_str)
        if not match:
            raise ValueError(f"Invalid semantic version: '{version_str}'")

        self.major = int(match.group(1))
        self.minor = int(match.group(2))
        self.patch = int(match.group(3))
        self.prerelease = match.group(4)  # Can be None
        self.metadata = match.group(5)   # Can be None
    
    @property
    def core(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self):
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.metadata:
            version += f"+{self.metadata}"
        return version

    def __repr__(self):
        return (
            f"Version(major={self.major}, minor={self.minor}, patch={self.patch}, "
            f"prerelease={self.prerelease!r}, metadata={self.metadata!r})"
        )


class PythonEnv:
    
    def __init__(self, env_path):
        self.env_path = Path(env_path)
        if platform.system() == 'Linux':
            self.env_path /= 'bin'
        self.executable = self.env_path / 'python'
    
    def exe(self) -> str:
        return str(self.executable)
    
    
class Wheel:
    
    def __init__(self, env:PythonEnv, libtorch_versions:list[Version]):
        self.python_env = env
        self.libtorch_versions = libtorch_versions
        
    def install_requirements(self):
        command = f"{self.python_env.exe()} -m pip install build numpy==1.26.0 pybind11 pybind11-stubgen"
        subprocess.run(command, check=True, shell=True)
         
    def build(self):
        print("[%] Installing requirements")
        self.install_requirements()
        
        for libtorch_version in self.libtorch_versions:
            # Install torch version
            command = f"{self.python_env.exe()} -m pip install torch=={libtorch_version.core}"
            subprocess.run(command, check=True, shell=True)
             
            # Build synapx and bindings
            sys.argv = ['None', 'bindings', '--py-exe', self.python_env.exe()]
            build_libsynapx.main()

            # Setup synapx
            setup_synapx.main(python_exe=self.python_env.exe())
            
        # Create wheel
        command = f"{self.python_env.exe()} -m build"
        subprocess.run(command, check=True, shell=True)

       
supported_python_versions = {
    PythonEnv(conda_envs_path/'py39'): [
        Version('2.0.0'), Version('2.1.0'), Version('2.2.0'), Version('2.3.0'), Version('2.4.0'), Version('2.5.0')
    ],
    PythonEnv(conda_envs_path/'py310'): [
        Version('2.0.0'), Version('2.1.0'), Version('2.2.0'), Version('2.3.0'), Version('2.4.0'), Version('2.5.0')
    ],
    PythonEnv(conda_envs_path/'py311'): [
        Version('2.0.0'), Version('2.1.0'), Version('2.2.0'), Version('2.3.0'), Version('2.4.0'), Version('2.5.0')
    ],
    PythonEnv(conda_envs_path/'py312'): [
        Version('2.2.0'), Version('2.3.0'), Version('2.4.0'), Version("2.5.0")
    ],
}
        
def main():
    t0 = time.time()
    for python_env, libtorch_versions in supported_python_versions.items():
        print(f"[%] Creating wheel for {python_env.exe()}")
        print("[%] Libtorch Versions Supported:")
        for v in libtorch_versions:
            print(f"     - {v.core}")
        
        # Remove lib folder
        if setup_synapx.package_lib_path.exists():
            shutil.rmtree(setup_synapx.package_lib_path)
            
        Wheel(python_env, libtorch_versions).build()
        print("[🗸] Wheel created successfully")
    
    print("\n[🗸] All wheels created successfully!")
    print(f"[%] Elapsed time: {time.time() - t0} s") 

if __name__ == '__main__':
    main()