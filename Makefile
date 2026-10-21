# GPU target (extra in pyproject.toml): gfx1200 or gfx1034. Detected from
# rocminfo when available, or set it: make install GPU=gfx1034. Empty = no ROCm.
GPU ?= $(shell rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' | head -1)
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

lint-strict:
	$(UV) flake8 .
	$(UV) mypy . --strict
