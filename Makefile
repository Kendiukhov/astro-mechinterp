.PHONY: help setup test figures paper clean

help:
	@echo "setup    - install the package and dev dependencies"
	@echo "test     - smoke tests (no model, no data required)"
	@echo "figures  - rebuild all paper figures from results/*.json"
	@echo "paper    - compile the manuscript (requires a TeX distribution)"
	@echo "clean    - remove build artefacts"

setup:
	pip install -e ".[dev]"

test:
	pytest -q

figures:
	python experiments/make_paper_figures.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && bibtex main >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && echo "paper/main.pdf built"

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache *.egg-info
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out
