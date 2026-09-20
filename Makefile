.PHONY: demo test lint

demo:
	python scripts/run_demo.py

test:
	python -m pytest tests -q

lint:
	ruff check ebro_mh scripts tests
