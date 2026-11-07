# Torch target (extra in pyproject.toml): cpu, gfx1200 or gfx1034. Detected
# from rocminfo when available, or set it: make install GPU=gfx1034. Falls
# back to cpu when no ROCm GPU is found.
GPU ?= $(or $(shell rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' | head -1),cpu)
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
endif

UV = uv run $(if $(GPU),--extra $(GPU))
PYTHON = $(UV) python

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync $(if $(GPU),--extra $(GPU))

run:
	$(if $(wildcard .env),UV_ENV_FILE=.env )$(PYTHON) -m src

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
