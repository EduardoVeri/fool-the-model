# Fool the Model: Developing Adversarial Perturbations Against DeepFake Classifiers

This repository contains the LaTeX source files for the report titled "Fool the Model: Developing Adversarial Perturbations Against DeepFake Classifiers". The report provides an overview of the project, including the motivation, methodology, results, and conclusions.

👉 **[View the compiled PDF report here](./main.pdf)**.

## Prerequisites

Ensure you have the following software installed on your system:
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- A text editor or LaTeX editor (e.g., Overleaf)

## Steps to Compile the Report

Run the following commands in sequence from the terminal or use the compile options in your LaTeX editor:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

- The first `pdflatex` command compiles the document and generates an auxiliary file (`.aux`).
- The `bibtex` command processes the bibliography file (`.bib`) and generates a bibliography file (`.bbl`).
- The second and third `pdflatex` commands are necessary to resolve references and ensure the bibliography is correctly included in the document.

**After successful compilation, the output PDF file (`main.pdf`) will be generated in the same directory.**

## Notes
- This project adheres to the SBC template for academic papers.
- Ensure that all required files (`main.tex`, `sbc-template.sty`, `sbc-template.bib`, `sbc.bst`) are in the same directory before compiling.