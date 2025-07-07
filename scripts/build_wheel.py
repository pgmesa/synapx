
import os
import sys
import time
import argparse
import platform
import subprocess
from pathlib import Path

project_path = Path(__file__).parent.parent
if str(project_path) not in sys.path:
    sys.path.append(str(project_path))

import toml
from packaging.requirements import Requirement
from packaging.version import Version

from synapx import synapx_lib_dir, torch_lib_dir


python_version = f"{sys.version_info.major}.{sys.version_info.minor}"


def get_minor_versions_in_range(start: Version, end: Version) -> list[str]:
    versions = []
    major = start.major
    minor = start.minor
    while (v := Version(f"{major}.{minor}.0")) <= end:
        versions.append(str(v))
        minor += 1
    return versions


def read_torch_versions(toml_path) -> list[str]:
    data = toml.load(toml_path)
    specifiers = []
    for dep in data["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "torch" and (
            req.marker is None or req.marker.evaluate({"python_version": python_version})
        ):
            specifiers.append(req.specifier)
    
    allowed = set()
    for spec in specifiers:
        # Find lowest and highest bounds
        min_ver = max_ver = None
        for s in spec:
            if s.operator in (">=", ">"):
                ver = Version(s.version)
                if min_ver is None or ver > min_ver:
                    min_ver = ver
            elif s.operator in ("<=", "<"):
                ver = Version(s.version)
                if max_ver is None or ver < max_ver:
                    max_ver = ver

        if min_ver and max_ver:
            allowed.update(get_minor_versions_in_range(min_ver, max_ver))

    return sorted(allowed, key=Version)


def main():
    parser = argparse.ArgumentParser(description="Build SynapX Wheel")
    parser.add_argument('--preset', required=False, type=str, help="CMake preset to use")
    
    args = parser.parse_args()
    
    preset = args.preset
    
    t0 = time.time()
    system = platform.system()
    libsynapx_dir = project_path / 'libsynapx'
    (libsynapx_dir / 'build').mkdir(exist_ok=True)
    toml_file_path = project_path / 'pyproject.toml'
    index_url = 'https://download.pytorch.org/whl/cpu/'

    torch_versions = read_torch_versions(toml_file_path)
    print(f"[%] Detected torch versions to build against: {torch_versions}")

    for i, v in enumerate(torch_versions):
        cmd = [sys.executable, '-m', 'pip', 'install', f'torch=={v}', f'--index-url={index_url}']
        subprocess.run(cmd, cwd=project_path, check=True)
        
        if preset is None:
            preset = 'runner-' + system.lower()
        
        subprocess.run(['make', 'rebuild', f'preset={preset}', 'target=install'], cwd=libsynapx_dir, check=True)
        if i == 0:
            subprocess.run([sys.executable, 'scripts/generate_pyi.py'], cwd=project_path, check=True)

        print(f"[%] Running tests...")
        subprocess.run([sys.executable, '-m', 'pytest', 'tests'], cwd=project_path, check=True)
    
    print(f"[%] Building wheel...")
    subprocess.run([sys.executable, '-m', 'build'], cwd=project_path, check=True)
    print("[OK] Wheel created successfully")

    if system == 'Linux':
        print("[%] Repairing Wheel...")

        # Add libsynapx.so and torch libraries to LD_LIBRARY_PATH
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        libs_dir = f"{synapx_lib_dir}:{torch_lib_dir}"
        new_ld_path = f"{libs_dir}:{current_ld_path}" if current_ld_path else str(libs_dir)
        print(f"Using LD_LIBRARY_PATH: {new_ld_path}")

        # Create environment with updated LD_LIBRARY_PATH
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = new_ld_path

        # Get wheel file name
        wheel_file = None
        wheel_dir = project_path / 'dist'
        for f in wheel_dir.iterdir():
            if f.name.endswith('linux_x86_64.whl'):
                wheel_file = f.relative_to(project_path)
                print(f"[%] Wheel found at '{wheel_file}'")
                break
        else:
            print(f"[!] No wheel found in {wheel_dir}")
            exit(-1)

        # Run auditwheel
        print("[%] Running auditwheel...")
        subprocess.run(
            ['auditwheel', 'repair', str(wheel_file), '-w', 'dist'], env=env, check=True
        )
        
        print(f"[%] Removing {wheel_file}...")
        os.remove(wheel_file)

        print("[OK] Wheel repaired successfully")

    print(f"[%] Elapsed time: {round(time.time() - t0, 2)} s")


if __name__ == '__main__':
    main()
