
import shutil
import argparse
import subprocess
from pathlib import Path

from scripts import setup_synapx


# Paths and configurations
ROOT_DIR = Path(__file__).parent.parent
TARGET_DIR = ROOT_DIR / "libsynapx"
BUILD_DIR = TARGET_DIR / "build"
CONFIGURATION = "Release"


# Helper functions
def run_command(command):
    """Run a shell command and check for errors."""
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def clean():
    """Clean the build directory."""
    print("Cleaning build directory...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"Removed {BUILD_DIR}")
    else:
        print(f"{BUILD_DIR} does not exist. Nothing to clean.")

def build_all():
    """Build the project."""
    clean()
    print("Building the project...")
    run_command([
        "cmake", "-S", str(TARGET_DIR), "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={CONFIGURATION}",
        "-DBUILD_CPP_TESTS=ON",
        "-DBUILD_PYTHON_BINDINGS=ON"
    ])
    run_command(["cmake", "--build", str(BUILD_DIR), "--config", CONFIGURATION])
    setup_synapx.main()

def build_python():
    """Build only the Python bindings."""
    clean()
    print("Building Python bindings...")
    run_command([
        "cmake", "-S", str(TARGET_DIR), "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={CONFIGURATION}",
        "-DBUILD_CPP_TESTS=OFF",
        "-DBUILD_PYTHON_BINDINGS=ON"
    ])
    run_command(["cmake", "--build", str(BUILD_DIR), "--config", CONFIGURATION])
    setup_synapx.main()

def build_tests():
    """Compile tests."""
    clean()
    print("Running tests...")
    run_command([
        "cmake", "-S", str(TARGET_DIR), "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={CONFIGURATION}",
        "-DBUILD_CPP_TESTS=ON",
        "-DBUILD_PYTHON_BINDINGS=OFF"
    ])
    run_command(["cmake", "--build", str(BUILD_DIR)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and manage the project.")
    parser.add_argument("task", choices=["all", "python", "tests", "clean"], help="Task to perform")
    args = parser.parse_args()

    tasks = {
        "all": build_all,
        "python": build_python,
        "tests": build_tests,
        "clean": clean,
    }

    nchars = 50
    print("-"*nchars)
    print("Building LibSynapX C++".center(nchars))
    print("-"*nchars)

    try:
        tasks[args.task]()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
