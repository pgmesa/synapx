# Variables
TARGET_DIR = .
BUILD_DIR = $(TARGET_DIR)\build
PROJECT_NAME = synapx_project.sln
CONFIGURATION = Release
PYD_OUTPUT_DIR = $(BUILD_DIR)\$(CONFIGURATION)
PYD_EXTENSION = *.pyd

CPP_COMPILER=g++

all: build run

build:
    # Generate Visual Studio solution files with cmake
	cmake -S . -B $(BUILD_DIR) -G "Visual Studio 17 2022" -A x64

    # Build the project using msbuild
	msbuild $(BUILD_DIR)\$(PROJECT_NAME) /p:Configuration=$(CONFIGURATION)

    # Copy the generated .pyd file from the build directory to the current directory
	cmd /C "copy $(PYD_OUTPUT_DIR)\$(PYD_EXTENSION) ."

    # Generate .pyi file
	pybind11-stubgen synapx_c --output .

run:
	python $(TARGET_DIR)\z_test.py

# Clean rule to remove build files and .pyd files
clean:
    # Remove the build directory
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
    # Remove the .pyd file from the current directory
	@if exist $(PYD_EXTENSION) del /q $(PYD_EXTENSION)