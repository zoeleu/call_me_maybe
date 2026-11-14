# 42 computers: we need to keep packages and the venv in ~/goinfre
# it's fast, and plentiful ephemeral storage, perfect for this purpose
GOINFRE := $(wildcard $(HOME)/goinfre)
ifneq ($(GOINFRE),)
G := $(GOINFRE)/call_me_maybe
export UV_CACHE_DIR ?= $(G)/uv-cache
export UV_PROJECT_ENVIRONMENT ?= $(G)/venv
export UV_PYTHON_INSTALL_DIR ?= $(G)/python
export UV_LINK_MODE ?= hardlink
export HF_HOME ?= $(G)/hf
export MYPY_CACHE_DIR ?= $(G)/mypy-cache
endif

UV = uv run
PYTHON = $(UV) python

.PHONY: install run debug clean lint lint-strict test bench

install:
	uv sync

run:
	$(if $(wildcard .env),UV_ENV_FILE=.env )$(PYTHON) -m src $(ARGS)

# Run every case in bench/cases/ through the program itself (one run per
# case: each case has its own function list), then grade the outputs.
bench:
	@mkdir -p data/output/xlam
	@for case_dir in bench/cases/*/; do \
		case=$$(basename $$case_dir); \
		echo "== $$case"; \
		$(if $(wildcard .env),UV_ENV_FILE=.env )$(PYTHON) -m src $(ARGS) \
			--functions_definition $$case_dir/functions_definition.json \
			--input $$case_dir/function_calling_tests.json \
			--output data/output/xlam/$$case.json \
			|| echo "run failed: $$case"; \
	done
	$(PYTHON) bench/grade.py data/output/xlam

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
		-not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf data/output

lint:
	$(UV) flake8 .
	$(UV) mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

test:
	$(UV) pytest

lint-strict:
	$(UV) flake8 .
	$(UV) mypy . --strict
