pipinstall:
	pip install -e .

uvsync:
	uv sync --index-strategy unsafe-best-match

# Format the source code using ruff
format:
	ruff format src

check:
	ruff check src

.PHONY: uvsync format pipinstall check
