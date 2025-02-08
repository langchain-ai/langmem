.PHONY: lint-docs format-docs build-docs serve-docs serve-clean-docs clean-docs codespell build-typedoc doctest

build-docs:
	uv run --with-editable . python -m mkdocs build --clean -f docs/mkdocs.yml --strict

serve-clean-docs: clean-docs
	uv run --with-editable . python -m mkdocs serve -c -f docs/mkdocs.yml --strict -w ./src/langmem

serve-docs: build-typedoc
	uv run --with-editable . python -m mkdocs serve -f docs/mkdocs.yml -w ./src/langmem -w README.md

## Run format against the project documentation.
format-docs:
	uv run ruff format docs/docs
	uv run ruff check --fix docs/docs

doctest:
	@echo "Starting langgraph server..."
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --no-browser > /dev/null 2>&1 & echo $$! > .langgraph.pid
	@echo "Waiting for server to start..."
	@sleep 2
	@echo "Running tests..."
	uv run --with-editable . python -m pytest tests/test_docstring_examples.py -vvv -n auto $(if $(k),-k "$(k)",) || (kill `cat .langgraph.pid` && rm .langgraph.pid && exit 1)
	@echo "Cleaning up server..."
	@kill `cat .langgraph.pid` && rm .langgraph.pid

doctest-watch:
	@echo "Starting langgraph server..."
	uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --no-browser > /dev/null 2>&1 & echo $$! > .langgraph.pid
	@echo "Waiting for server to start..."
	@sleep 2
	@echo "Starting test watcher..."
	PYTHONPATH=src uv run --with-editable . ptw tests/test_docstring_examples.py -- -vvv --last-failed --new-first $(if $(k),-k='$(k)',) || (kill `cat .langgraph.pid` && rm .langgraph.pid && exit 1)
	@echo "Cleaning up server..."
	@kill `cat .langgraph.pid` && rm .langgraph.pid


format:
	uv run ruff format ./src
	uv run ruff check --fix ./src

lint:
	uv run ruff format --check ./src
	uv run ruff check ./src

# Check the docs for linting violations
lint-docs:
	uv run ruff format --check docs/docs
	uv run ruff check docs/docs

	uv run ruff format --check docs/docs
	uv run ruff check docs/docs
