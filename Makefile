# OS detection
ifeq ($(OS),Windows_NT)
    # Windows settings
    RM = if exist $(BUILD_DIR) rmdir /s /q
    PYTHON = python
    CONFIGURATION = Release
    TARGET_DIR = .\libsynapx
    BUILD_DIR = $(TARGET_DIR)\build
    PYD_OUTPUT_DIR = $(BUILD_DIR)\$(PROJECT_NAME)\$(CONFIGURATION)
    PYD_DESTINATION = $(TARGET_DIR)\$(PROJECT_NAME)
    SETUP_SCRIPT = scripts\setup_synapx.py
else
    # Linux/Mac settings
    RM = rm -rf
    PYTHON = python
    CONFIGURATION = Release
    TARGET_DIR = ./libsynapx
    BUILD_DIR = $(TARGET_DIR)/build
    PYD_OUTPUT_DIR = $(BUILD_DIR)/$(PROJECT_NAME)/$(CONFIGURATION)
    PYD_DESTINATION = $(TARGET_DIR)/$(PROJECT_NAME)
    SETUP_SCRIPT = scripts/setup_synapx.py
endif

# Compiler
CPP_COMPILER = g++

# Default target
all: build

# Build the project
build: clean
	@echo "Building the project..."
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=ON \
		-DBUILD_PYTHON_BINDINGS=ON
	cmake --build $(BUILD_DIR) --config $(CONFIGURATION)
	$(PYTHON) $(SETUP_SCRIPT)

# Python-specific build
python: clean
	@echo "Building Python bindings..."
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=OFF \
		-DBUILD_PYTHON_BINDINGS=ON
	cmake --build $(BUILD_DIR) --config $(CONFIGURATION)
	$(PYTHON) $(SETUP_SCRIPT)

# Run tests
tests: clean
	@echo "Running tests..."
	cmake -S $(TARGET_DIR) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CONFIGURATION) \
		-DBUILD_CPP_TESTS=ON \
		-DBUILD_PYTHON_BINDINGS=OFF
	cmake --build $(BUILD_DIR)

# Clean rule to remove build files
clean:
	@echo "Cleaning build directory..."
	$(RM) $(BUILD_DIR)
