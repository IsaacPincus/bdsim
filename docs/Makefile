# Build the bdsim documentation.
#
#   make html     -> _build/html/index.html
#   make theory   -> theory.pdf (the mathematics)
#   make clean
#
# Sphinx is invoked as `python -m sphinx` rather than the `sphinx-build` script.
# That uses whichever interpreter is active, so it works when the venv's scripts
# directory is not on PATH (a common `pip install --user` outcome) and cannot pick
# up a system-wide sphinx-build belonging to a different Python.

PYTHON  ?= python
BUILDDIR = _build
SOURCEDIR = .

.PHONY: help html theory all clean check-sphinx

help:
	@echo "make html    build the HTML documentation"
	@echo "make theory  build theory.pdf"
	@echo "make all     both"
	@echo "make clean   remove build artefacts"
	@echo ""
	@echo "Documentation dependencies (separate from the runtime ones):"
	@echo "  pip install sphinx myst-parser sphinx-rtd-theme"

check-sphinx:
	@$(PYTHON) -c "import sphinx, myst_parser, sphinx_rtd_theme" 2>/dev/null || { \
	  echo ""; \
	  echo "The documentation dependencies are missing from this environment."; \
	  echo "  interpreter: $$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	  echo ""; \
	  echo "  pip install sphinx myst-parser sphinx-rtd-theme"; \
	  echo ""; \
	  echo "(They are deliberately not runtime dependencies of bdsim.)"; \
	  echo ""; \
	  exit 1; }

html: check-sphinx
	$(PYTHON) -m sphinx -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "-> $(BUILDDIR)/html/index.html"

theory:
	@command -v pdflatex >/dev/null || { \
	  echo "pdflatex not found; install a LaTeX distribution (e.g. texlive-latex-recommended)"; \
	  exit 1; }
	pdflatex -interaction=nonstopmode theory.tex >/dev/null
	pdflatex -interaction=nonstopmode theory.tex >/dev/null
	@rm -f theory.aux theory.log theory.toc theory.out
	@echo "-> theory.pdf"

all: html theory

clean:
	rm -rf $(BUILDDIR) theory.aux theory.log theory.toc theory.out
