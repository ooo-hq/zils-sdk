PYTHON ?= python3
PLATFORM_DIR ?= ../zils-platform
.PHONY: check check-integration
check:
	$(PYTHON) -m ruff check packages/python packages/cli
	$(PYTHON) -m ruff format --check packages/python packages/cli
	npm --prefix packages/typescript test
	$(PYTHON) -m unittest discover -s packages/python/tests -v

check-integration:
	test -f "$(PLATFORM_DIR)/tests/test_api.py"
	npm --prefix packages/typescript run build
	ZILS_TEST_PLATFORM=1 PYTHONPATH="$(abspath $(PLATFORM_DIR)):$(PYTHONPATH)" $(PYTHON) -m unittest discover -s packages/python/tests -v
