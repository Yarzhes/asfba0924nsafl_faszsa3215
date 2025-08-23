.PHONY: help setup test run-live clean

help:
	@echo "Makefile for Ultra-Signals"
	@echo ""
	@echo "Usage:"
	@echo "    make setup          - Create virtual environment and install dependencies"
	@echo "    make test           - Run all tests with pytest"
	@echo "    make run-live       - Run the realtime trading runner"
	@echo "    make clean          - Remove virtual environment and temporary files"

setup:
	@echo "---> Setting up virtual environment..."
	python -m venv .venv
	@echo "---> Activating virtual environment and installing dependencies..."
	@/bin/bash -c "source .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt"
	@echo ""
	@echo "Setup complete. To activate the venv, run: source .venv/bin/activate"

test:
	@echo "---> Running tests..."
	@/bin/bash -c "source .venv/bin/activate; pytest"

run-live:
	@echo "---> Running live application..."
	@/bin/bash -c "source .venv/bin/activate; python -m apps.realtime_runner --config settings.yaml"

clean:
	@echo "---> Cleaning up..."
	rm -rf .venv
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -rf .pytest_cache