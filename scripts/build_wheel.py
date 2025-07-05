
import sys
import time
import platform
import subprocess
from pathlib import Path

import toml
from packaging.requirements import Requirement
from packaging.version import Version


project_path = Path(__file__).parent.parent
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
    t0 = time.time()
    libsynapx_dir = project_path / 'libsynapx'
    (libsynapx_dir / 'build').mkdir(exist_ok=True)
    toml_file_path = project_path / 'pyproject.toml'
    index_url = 'https://download.pytorch.org/whl/cpu/'

    torch_versions = read_torch_versions(toml_file_path)
    print(f"[+] Detected torch versions to build against: {torch_versions}")

    for i, v in enumerate(torch_versions):
        cmd = [sys.executable, '-m', 'pip', 'install', f'torch=={v}', f'--index-url={index_url}']
        subprocess.run(cmd, cwd=project_path, check=True)
        
        preset = 'runner-' + platform.system().lower()
        subprocess.run(['make', 'rebuild', f'preset={preset}', 'target=install'], cwd=libsynapx_dir, check=True)
        if i == 0:
            subprocess.run([sys.executable, 'scripts/generate_pyi.py'], cwd=project_path, check=True)

        print(f"[%] Running tests...")
        subprocess.run([sys.executable, '-m', 'pytest', 'tests'], cwd=project_path, check=True)
    
    print(f"[%] Building wheel...")
    subprocess.run([sys.executable, '-m', 'build'], cwd=project_path, check=True)
    print("[✓] Wheel created successfully")
    print(f"[✓] Elapsed time: {round(time.time() - t0, 2)} s")


if __name__ == '__main__':
    main()
