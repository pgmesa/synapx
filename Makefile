# Variables
TARGET_DIR = .
BUILD_DIR = $(TARGET_DIR)\build
PROJECT_NAME = synapx
OUTPUT_NAME = synapx_c
SOLUTION_FILE = $(PROJECT_NAME)_project.sln
CONFIGURATION = Release
PYD_OUTPUT_DIR = $(BUILD_DIR)\$(PROJECT_NAME)\$(CONFIGURATION)
PYD_EXTENSION = *.pyd
PYD_DESTINATION = $(TARGET_DIR)\$(PROJECT_NAME)

CPP_COMPILER=g++

all: clean build

build:
    # Generate Visual Studio solution files with cmake
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

    # Build the project using msbuild
	msbuild $(BUILD_DIR)\$(SOLUTION_FILE) /p:Configuration=$(CONFIGURATION)

    # Copy the generated .pyd file from the build directory to the current directory
	cmd /C "copy $(PYD_OUTPUT_DIR)\$(PYD_EXTENSION) ."

    # Generate .pyi file
	pybind11-stubgen $(OUTPUT_NAME) --output .
	cmd /C "move *.pyd .\synapx"
	cmd /C "move *.pyi .\synapx"

# Clean rule to remove build files and .pyd files
clean:
    # Remove the build directory
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    # Remove the .pyd file from the current directory
	@if exist $(PYD_DESTINATION) del /q $(PYD_DESTINATION)\$(PYD_EXTENSION)