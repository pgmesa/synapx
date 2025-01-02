# Shortcuts for building the project

# OS detection
ifeq ($(OS),Windows_NT)
    # Windows settings
    BUILD_SCRIPT = scripts\build_libsynapx.py
else
    # Linux/Mac settings
    BUILD_SCRIPT = scripts/build_libsynapx.py
endif

# Python executable
PYTHON = python

# -- Build libsynapx --
# All 
all: 
	@echo "Creating Python bindings and compiling tests..."
	$(PYTHON) $(BUILD_SCRIPT) all

# Create python bindings
python: 
	@echo "Creating Python bindings..."
	$(PYTHON) $(BUILD_SCRIPT) "python"

# Compile tests
tests: 
	@echo "Compiling tests..."
	$(PYTHON) $(BUILD_SCRIPT) tests

# Clean rule to remove build files
clean: 
	@echo "Cleaning build files..."
	$(PYTHON) $(BUILD_SCRIPT) clean
