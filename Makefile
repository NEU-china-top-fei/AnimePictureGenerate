PYTHON ?= python3

.PHONY: test check export

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test
	$(PYTHON) scripts/check_repository.py

export:
	$(PYTHON) scripts/export_repository.py
