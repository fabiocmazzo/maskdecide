.DEFAULT_GOAL := help

UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000
TEST_ARGS ?=

.PHONY: help install serve test test-smoke

help:
	@echo "MaskDecide"
	@echo "  make install     Install locked dependencies with uv"
	@echo "  make serve       Start the API server (requires CUDA)"
	@echo "  make test        Run local tests (no GPU or model download)"
	@echo "  make test-smoke  Check a running API server and decision accuracy"
	@echo "Options: HOST=127.0.0.1 PORT=8000 TEST_ARGS='--repeat 3'"

install:
	$(UV) sync --locked

serve:
	$(UV) run --locked maskdecide --host "$(HOST)" --port "$(PORT)"

test:
	$(UV) run --locked python -m unittest discover -s tests -v

test-smoke:
	$(UV) run --locked python tests/test_jev_api_smoke.py --url "http://127.0.0.1:$(PORT)" $(TEST_ARGS)
