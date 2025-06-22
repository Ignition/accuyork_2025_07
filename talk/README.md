# LaTeX Beamer Presentation

A clean and professional LaTeX Beamer presentation template.

## Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- `pdflatex` (included in most LaTeX distributions)

## Getting Started

1. Clone this repository
2. Edit the `main.tex` file to customize your presentation
3. Build the presentation

## Building the Presentation

### Using Make (recommended)

```bash
make           # Build the presentation
make view     # Build and open the presentation
make clean    # Clean auxiliary files
make cleanall # Clean all generated files including PDF
```

### Manual Build

```bash
pdflatex main.tex
pdflatex main.tex  # Run twice to resolve references
```

## Project Structure

- `main.tex` - Main presentation file
- `Makefile` - Build automation
- `README.md` - This file

## Customization

- Change the theme in `main.tex` by modifying `\usetheme{Madrid}`
- Add images to the `images/` directory and include them using `\includegraphics`
- Customize colors using `\usecolortheme`

## License

This project is open source and available under the [MIT License](LICENSE).
