# Variables
TARGET_DIR = .\libsynapx
BUILD_DIR = $(TARGET_DIR)\build
CONFIGURATION = Release
PYD_OUTPUT_DIR = $(BUILD_DIR)\$(PROJECT_NAME)\$(CONFIGURATION)
PYD_DESTINATION = $(TARGET_DIR)\$(PROJECT_NAME)

CPP_COMPILER=g++

all: build

build:
	make clean
    # Generate Visual Studio solution files with cmake
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=ON \
		-DBUILD_PYTHON_BINDINGS=ON

    # Build the project
	cmake --build $(BUILD_DIR) --config $(CONFIGURATION)
	
    # Setup Python Package
	python scripts\setup_synapx_win.py

python:
	make clean
    # Generate Visual Studio solution files with cmake
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=OFF \
		-DBUILD_PYTHON_BINDINGS=ON

    # Build the project
	cmake --build $(BUILD_DIR) --config $(CONFIGURATION)
	
    # Setup Python Package
	python scripts\setup_synapx_win.py

tests:
	make clean
    # Generate Visual Studio solution files with cmake
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=ON \
		-DBUILD_PYTHON_BINDINGS=OFF

    # Build the project
	cmake --build $(BUILD_DIR)

# Clean rule to remove build files and .pyd files
clean:
    # Remove the build directory
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)