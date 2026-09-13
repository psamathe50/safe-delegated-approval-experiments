PYTHON ?= python3
.PHONY: all experiments verify setup

all: experiments verify

setup:
	$(PYTHON) -m pip install -r requirements.txt

experiments:
	$(PYTHON) run.py

verify:
	$(PYTHON) verify.py
