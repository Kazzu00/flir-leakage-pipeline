.PHONY: install test lint format clean

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	uv cache clean
