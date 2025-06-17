# Makefile for FL4000 Project

PYTHON := .venv/bin/python
PYTEST := .venv/bin/python -m pytest

export PYTHONPATH := src

.PHONY: simulate test visualize

simulate:
	$(PYTHON) src/federated/simulation.py

test:
	$(PYTEST) tests --maxfail=3 --disable-warnings -v

visualize:
	$(PYTHON) scripts/visualize_clipping_effect.py
