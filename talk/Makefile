.PHONY: all clean

# Main file name (without .tex extension)
MAIN = main

# Compiler and flags
LATEX = pdflatex
LATEXFLAGS = -shell-escape -interaction=nonstopmode -file-line-error -output-directory=build

# Output PDF name
PDF = build/$(MAIN).pdf

all: $(PDF)


$(PDF): $(MAIN).tex
	mkdir -p build
	$(LATEX) $(LATEXFLAGS) $<
	$(LATEX) $(LATEXFLAGS) $< # Run twice to resolve references

clean:
	rm -rf build

.PHONY: view
view: $(PDF)
	xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || echo "Could not open PDF viewer"
