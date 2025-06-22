.PHONY: all clean venv install-deps latex-clean

# Main file name (without .tex extension)
MAIN = main

# Python virtual environment
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Compiler and flags
LATEX = pdflatex
LATEXFLAGS = -shell-escape -interaction=nonstopmode -file-line-error -output-directory=build

# Output PDF name
PDF = build/$(MAIN).pdf

# Default target
all: $(PDF)

# Set up virtual environment
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV_DIR)/bin/activate

# Install/update dependencies
install-deps: venv
	$(PIP) install -r requirements.txt

# Verify Pygments is installed and at correct version
check-pygments: venv
	@echo "Checking Pygments version..."
	@$(PYTHON) -m pip show Pygments > /dev/null || \
		( echo "✗ Pygments is not installed." && exit 1 )
	@$(PYTHON) -c "import sys; from packaging import version; import pygments; \
		required = version.parse('2.19.2'); actual = version.parse(pygments.__version__); \
		assert actual >= required, f'✗ Pygments {actual} is too old, need >= {required}'; \
		print(f'✓ Pygments {actual} meets version requirements')"

# Build the PDF
$(PDF): $(MAIN).tex | check-pygments
	mkdir -p build
	# Ensure LaTeX can find the Python executable from our venv
	PYTHONPATH=$(VENV_DIR)/lib/python*/site-packages \
	PYTHONEXECUTABLE=$(PYTHON) \
	TEXINPUTS=.:$(shell pwd): \
	$(LATEX) $(LATEXFLAGS) $<
	# Run twice to resolve references
	PYTHONPATH=$(VENV_DIR)/lib/python*/site-packages \
	PYTHONEXECUTABLE=$(PYTHON) \
	TEXINPUTS=.:$(shell pwd): \
	$(LATEX) $(LATEXFLAGS) $<

# Clean build artifacts
clean: latex-clean
	rm -rf $(VENV_DIR)

# Clean LaTeX build files
latex-clean:
	rm -rf build

# View the PDF
.PHONY: view
view: $(PDF)
	xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || echo "Could not open PDF viewer"
