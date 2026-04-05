.PHONY: clean

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

comparison:
	uv run main.py comp

comparison10x:
	uv run main.py comp --runs 10

depth:
	uv run depth.py
